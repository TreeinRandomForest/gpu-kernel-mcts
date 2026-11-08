"""Conservative source-edit guard for opt-in Triton H100 actions."""

from __future__ import annotations

import ast
import copy


TRAVERSAL_ACTION = "single_cta_tile_traversal"
COALESCED_LAYOUT_ACTION = "gluon_mma_v2_coalesced_layout"
SERIAL_WGMMA_ACTION = "single_cta_gluon_tma_wgmma_serial_guarded"
PIPELINED_WGMMA_ACTION = "single_cta_gluon_tma_wgmma_wait_one"
TRITON_POINTER_LOOP_ACTION = "triton_pointer_advance_tile_128"
TRITON_WIDE_TILE_ACTION = "triton_widen_bn_256_eight_warps"
TRITON_CONSTEXPR_ACTION = "triton_specialize_gemm_dims"
COALESCED_ORDERS = {"a_layout": [1, 0], "b_layout": [0, 1],
                    "out_layout": [1, 0]}


def _assigned_name(statement: ast.stmt) -> str | None:
    if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)):
        return statement.targets[0].id
    if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
        return statement.target.id
    return None


def _function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    return next((item for item in tree.body
                 if isinstance(item, ast.FunctionDef) and item.name == name), None)


def _index(body: list[ast.stmt], name: str) -> int | None:
    return next((index for index, item in enumerate(body)
                 if _assigned_name(item) == name), None)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _call_name(node.value) + "." + node.attr
    return ""


def _launched_kernel(tree: ast.Module) -> tuple[ast.FunctionDef, ast.FunctionDef] | None:
    launch = _function(tree, "launch")
    if launch is None:
        return None
    names = [node.value.func.value.id for node in ast.walk(launch)
             if isinstance(node, ast.Return) and isinstance(node.value, ast.Call)
             and isinstance(node.value.func, ast.Subscript)
             and isinstance(node.value.func.value, ast.Name)]
    kernel = _function(tree, names[0]) if len(names) == 1 else None
    return (launch, kernel) if kernel is not None else None


def _launch_literal(launch: ast.FunctionDef, name: str) -> ast.Constant | None:
    for statement in launch.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target, value = statement.targets[0], statement.value
        if isinstance(target, ast.Name) and target.id == name and isinstance(value, ast.Constant):
            return value
        if isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple):
            for item, literal in zip(target.elts, value.elts):
                if isinstance(item, ast.Name) and item.id == name and isinstance(literal, ast.Constant):
                    return literal
    return None


def _pointer_advance(kernel: ast.FunctionDef, name: str) -> bool:
    return any(isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name)
               and node.target.id == name and isinstance(node.op, ast.Add)
               and isinstance(node.value, ast.Name) and node.value.id == "BK"
               for loop in ast.walk(kernel) if isinstance(loop, ast.For)
               for statement in loop.body for node in ast.walk(statement))


def triton_pointer_loop_ready(source: str) -> bool:
    """Recognize the local pure-Triton pointer-loop family, not its performance."""
    try:
        parsed = _launched_kernel(ast.parse(source))
    except SyntaxError:
        return False
    if parsed is None:
        return False
    launch, kernel = parsed
    bn = _launch_literal(launch, "BN")
    return (_pointer_advance(kernel, "a_ptrs")
            and _pointer_advance(kernel, "b_ptrs")
            and bn is not None and type(bn.value) is int)


def triton_wide_tile_actionable(source: str) -> bool:
    if not triton_pointer_loop_ready(source):
        return False
    parsed = _launched_kernel(ast.parse(source))
    assert parsed is not None
    launch, _ = parsed
    values = {name: _launch_literal(launch, name) for name in ("BM", "BN", "BK")}
    if not all(values[name] is not None and values[name].value == expected
               for name, expected in (("BM", 128), ("BN", 128), ("BK", 64))):
        return False
    return any(all(any(keyword.arg == key and isinstance(keyword.value, ast.Constant)
                       and keyword.value.value == value for keyword in call.keywords)
                   for key, value in (("num_warps", 8), ("num_stages", 3)))
               for call in ast.walk(launch)
               if isinstance(call, ast.Call) and isinstance(call.func, ast.Subscript))


def triton_constexpr_actionable(source: str) -> bool:
    """Limit dimension specialization to the measured wide pointer-loop family."""
    try:
        parsed = _launched_kernel(ast.parse(source))
    except SyntaxError:
        return False
    if parsed is None or not triton_pointer_loop_ready(source):
        return False
    launch, kernel = parsed
    values = {name: _launch_literal(launch, name) for name in ("BM", "BN", "BK")}
    if not all(values[name] is not None and values[name].value == expected
               for name, expected in (("BM", 128), ("BN", 256), ("BK", 64))):
        return False
    dimensions = [arg for arg in kernel.args.args if arg.arg in {"M", "N", "K"}]
    if len(dimensions) != 3 or any(arg.annotation is not None for arg in dimensions):
        return False
    return any(all(any(keyword.arg == key and isinstance(keyword.value, ast.Constant)
                       and keyword.value.value == value for keyword in call.keywords)
                   for key, value in (("num_warps", 8), ("num_stages", 3)))
               for call in ast.walk(launch)
               if isinstance(call, ast.Call) and isinstance(call.func, ast.Subscript))


def triton_constexpr_mismatch(parent_source: str, candidate_source: str) -> str | None:
    """Allow only M/N/K kernel-parameter annotations to become tl.constexpr."""
    try:
        parent, candidate = ast.parse(parent_source), ast.parse(candidate_source)
    except SyntaxError:
        return "source is not valid Python"
    p_pair, c_pair = _launched_kernel(parent), _launched_kernel(candidate)
    if p_pair is None or c_pair is None:
        return "both sources need a named JIT kernel returned by launch"
    p_dims = {arg.arg: arg for arg in p_pair[1].args.args if arg.arg in {"M", "N", "K"}}
    c_dims = {arg.arg: arg for arg in c_pair[1].args.args if arg.arg in {"M", "N", "K"}}
    if (set(p_dims) != {"M", "N", "K"} or set(c_dims) != set(p_dims)
            or any(p_dims[name].annotation is not None
                   or _call_name(c_dims[name].annotation) != "tl.constexpr"
                   for name in p_dims)):
        return "annotate exactly kernel parameters M, N, and K with tl.constexpr"
    for arg in c_dims.values():
        arg.annotation = None
    if ast.dump(parent, include_attributes=False) != ast.dump(candidate, include_attributes=False):
        return "change only the M/N/K constexpr annotations; preserve all other source"
    return None


def triton_pointer_loop_mismatch(parent_source: str, candidate_source: str) -> str | None:
    """Check the bounded root-to-pointer-loop action before normal evaluation."""
    del parent_source
    try:
        tree = ast.parse(candidate_source)
    except SyntaxError:
        return "source is not valid Python"
    parsed = _launched_kernel(tree)
    if parsed is None:
        return "launch must return one named Triton JIT kernel"
    launch, kernel = parsed
    issues: list[str] = []
    if not any(_call_name(item) == "triton.jit" for item in kernel.decorator_list):
        issues.append("keep a @triton.jit kernel")
    for name, expected in (("BM", 128), ("BN", 128), ("BK", 64)):
        literal = _launch_literal(launch, name)
        if literal is None or literal.value != expected:
            issues.append(f"set launch {name}={expected} as a literal")
    launches = [node for node in ast.walk(launch) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Subscript)]
    if not any(all(any(keyword.arg == key and isinstance(keyword.value, ast.Constant)
                       and keyword.value.value == value for keyword in call.keywords)
                   for key, value in (("num_warps", 8), ("num_stages", 3)))
               for call in launches):
        issues.append("launch with num_warps=8 and num_stages=3")
    if not triton_pointer_loop_ready(candidate_source):
        issues.append("advance both a_ptrs and b_ptrs by BK inside the K loop")
    dots = [node for node in ast.walk(kernel) if isinstance(node, ast.Call)
            and _call_name(node.func) == "tl.dot"]
    if not any(len(call.args) == 3 for call in dots):
        issues.append("accumulate with acc = tl.dot(a, b, acc)")
    program_axes = {node.args[0].value for node in ast.walk(kernel)
                    if isinstance(node, ast.Call) and _call_name(node.func) == "tl.program_id"
                    and node.args and isinstance(node.args[0], ast.Constant)}
    if program_axes != {0, 1}:
        issues.append("keep a two-dimensional CTA grid with tl.program_id(0) and (1)")
    return "; ".join(issues) if issues else None


def triton_wide_tile_mismatch(parent_source: str, candidate_source: str) -> str | None:
    """Allow only BN 128→256 in the measured eight-warp pointer-loop family."""
    try:
        parent, candidate = ast.parse(parent_source), ast.parse(candidate_source)
    except SyntaxError:
        return "source is not valid Python"
    p_pair, c_pair = _launched_kernel(parent), _launched_kernel(candidate)
    if p_pair is None or c_pair is None:
        return "both sources need a named JIT kernel returned by launch"
    p_bn = _launch_literal(p_pair[0], "BN")
    c_bn = _launch_literal(c_pair[0], "BN")
    if p_bn is None or p_bn.value != 128 or c_bn is None or c_bn.value != 256:
        return "change only the launch BN literal from 128 to 256"
    c_bn.value = 128
    if ast.dump(parent, include_attributes=False) != ast.dump(candidate, include_attributes=False):
        return "change only BN; preserve the pointer loop, eight warps, grid, masks, and all other source"
    return None


def _blocked_order(call: ast.AST | None) -> ast.AST | None:
    if not isinstance(call, ast.Call) or _call_name(call.func) != "gl.BlockedLayout":
        return None
    keywords = [item for item in call.keywords if item.arg == "order"]
    if len(call.args) == 4 and not keywords:
        return call.args[3]
    if len(call.args) <= 3 and len(keywords) == 1:
        return keywords[0].value
    return None


def _set_blocked_order(call: ast.Call, value: ast.AST) -> None:
    if len(call.args) == 4:
        call.args[3] = value
    else:
        next(item for item in call.keywords if item.arg == "order").value = value


def traversal_edit_mismatch(parent_source: str, candidate_source: str) -> str | None:
    """Allow only a grouped program-ID mapping and its one-dimensional grid.

    The static guard does not prove bijection; ordinary correctness remains required.
    """
    try:
        parent = ast.parse(parent_source)
        candidate = ast.parse(candidate_source)
    except SyntaxError:
        return "source is not valid Python"
    if ast.get_docstring(parent) != ast.get_docstring(candidate):
        return "preserve the top-level module docstring exactly"
    p_gemm, c_gemm = _function(parent, "gemm"), _function(candidate, "gemm")
    p_launch, c_launch = _function(parent, "launch"), _function(candidate, "launch")
    if None in (p_gemm, c_gemm, p_launch, c_launch):
        return "gemm and launch functions are required"
    assert p_gemm is not None and c_gemm is not None
    assert p_launch is not None and c_launch is not None
    p_start, p_end = _index(p_gemm.body, "pid_m"), _index(p_gemm.body, "off_m")
    c_end = _index(c_gemm.body, "off_m")
    p_grid, c_grid = _index(p_launch.body, "grid"), _index(c_launch.body, "grid")
    if None in (p_start, p_end, c_end, p_grid, c_grid):
        return "expected program-ID and launch-grid sites are missing"
    assert p_start is not None and p_end is not None and c_end is not None
    assert p_grid is not None and c_grid is not None
    if p_end <= p_start or c_end <= p_start:
        return "program-ID mapping is empty"
    middle = c_gemm.body[p_start:c_end]
    if not 3 <= len(middle) <= 20 or any(
        not isinstance(item, (ast.Assign, ast.AnnAssign)) for item in middle
    ):
        return "program-ID mapping must be bounded assignments"
    if sum(_assigned_name(item) == "pid_m" for item in middle) != 1 or sum(
        _assigned_name(item) == "pid_n" for item in middle
    ) != 1:
        return "mapping must assign pid_m and pid_n exactly once"
    group_values = [item.value for item in middle if _assigned_name(item) == "GROUP_M"]
    if len(group_values) != 1 or not isinstance(group_values[0], ast.Constant) or (
        type(group_values[0].value) is not int or group_values[0].value not in (2, 4, 8, 16)
    ):
        return "mapping must choose GROUP_M from 2, 4, 8, or 16"
    if not any(isinstance(node, ast.Name) and node.id == "GROUP_M"
               and isinstance(node.ctx, ast.Load)
               for item in middle for node in ast.walk(item)):
        return "mapping must use GROUP_M"
    if any(isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
           for item in middle for node in ast.walk(item)):
        return "mapping tile counts and indices must use integer // division"
    program_ids = [node for item in middle for node in ast.walk(item)
                   if isinstance(node, ast.Call) and _call_name(node.func) == "gl.program_id"]
    if (len(program_ids) != 1 or len(program_ids[0].args) != 1
            or not isinstance(program_ids[0].args[0], ast.Constant)
            or program_ids[0].args[0].value != 0):
        return "mapping must use only gl.program_id(0)"
    grid_statement = c_launch.body[c_grid]
    if (not isinstance(grid_statement, ast.Assign)
            or not isinstance(grid_statement.value, ast.Tuple)
            or len(grid_statement.value.elts) != 1):
        return "launch grid must be one-dimensional"
    if any(_assigned_name(item) == "grid" for item in c_launch.body[:c_grid]):
        return "launch grid must be assigned once"

    normalized_parent = copy.deepcopy(parent)
    normalized_candidate = copy.deepcopy(candidate)
    for tree, start, end, grid_index in (
        (normalized_parent, p_start, p_end, p_grid),
        (normalized_candidate, p_start, c_end, c_grid),
    ):
        gemm = _function(tree, "gemm")
        launch = _function(tree, "launch")
        assert gemm is not None and launch is not None
        gemm.body[start:end] = [ast.Pass()]
        grid = launch.body[grid_index]
        assert isinstance(grid, ast.Assign)
        grid.value = ast.Constant(value=None)
    if ast.dump(normalized_parent, include_attributes=False) != ast.dump(
        normalized_candidate, include_attributes=False
    ):
        return "candidate changed source outside program-ID mapping or launch grid"
    return None


def traversal_group_m(source: str) -> int | None:
    """Read a literal grouped-traversal size for a bounded sibling summary."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    gemm = _function(tree, "gemm")
    if gemm is None:
        return None
    values = [item.value for item in gemm.body if _assigned_name(item) == "GROUP_M"]
    if len(values) != 1 or not isinstance(values[0], ast.Constant):
        return None
    value = values[0].value
    return value if type(value) is int else None


def coalesced_layout_edit_mismatch(parent_source: str, candidate_source: str) -> str | None:
    """Admit only A, B, and C BlockedLayout axis-order edits for this GEMM ABI."""
    try:
        parent = ast.parse(parent_source)
        candidate = ast.parse(candidate_source)
    except SyntaxError:
        return "source is not valid Python"
    p_gemm, c_gemm = _function(parent, "gemm"), _function(candidate, "gemm")
    if p_gemm is None or c_gemm is None:
        return "gemm function is required"
    changed = 0
    for name, order in COALESCED_ORDERS.items():
        p_index, c_index = _index(p_gemm.body, name), _index(c_gemm.body, name)
        if p_index is None or c_index is None:
            return f"{name} assignment is required"
        p_stmt, c_stmt = p_gemm.body[p_index], c_gemm.body[c_index]
        p_value = p_stmt.value if isinstance(p_stmt, (ast.Assign, ast.AnnAssign)) else None
        c_value = c_stmt.value if isinstance(c_stmt, (ast.Assign, ast.AnnAssign)) else None
        if _blocked_order(p_value) is None or _blocked_order(c_value) is None:
            return f"{name} must be a gl.BlockedLayout with an axis order"
        assert isinstance(p_value, ast.Call) and isinstance(c_value, ast.Call)
        p_order, c_order = _blocked_order(p_value), _blocked_order(c_value)
        assert p_order is not None and c_order is not None
        if ast.dump(c_order) != ast.dump(ast.List(
            elts=[ast.Constant(item) for item in order], ctx=ast.Load())):
            return f"{name} order must be {order} for the GEMM tensor strides"
        changed += ast.dump(p_order) != ast.dump(c_order)
        _set_blocked_order(p_value, ast.Constant(None))
        _set_blocked_order(c_value, ast.Constant(None))
    if changed == 0:
        return "candidate did not change a layout axis order"
    if ast.dump(parent, include_attributes=False) != ast.dump(
        candidate, include_attributes=False
    ):
        return "candidate changed source outside A/B/C BlockedLayout axis orders"
    return None


def coalesced_layout_actionable(source: str) -> bool:
    """Offer the bounded action only when its three known layout sites exist."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    gemm = _function(tree, "gemm")
    if gemm is None:
        return False
    needs_change = False
    for name, order in COALESCED_ORDERS.items():
        index = _index(gemm.body, name)
        if index is None:
            return False
        statement = gemm.body[index]
        value = statement.value if isinstance(statement, (ast.Assign, ast.AnnAssign)) else None
        order_node = _blocked_order(value)
        if order_node is None:
            return False
        try:
            actual = ast.literal_eval(order_node)
        except (ValueError, TypeError, SyntaxError):
            return False
        needs_change |= actual != order
    return needs_change


def serial_wgmma_api_mismatch(parent_source: str, candidate_source: str) -> str | None:
    """Report known invalid API forms for the opt-in serial Hopper action.

    This checks a narrow API contract, not synchronization or GEMM correctness.
    """
    del parent_source
    try:
        tree = ast.parse(candidate_source)
    except SyntaxError:
        return "source is not valid Python"
    launch = _function(tree, "launch")
    if launch is None:
        return "launch(A,B,C,M,N,K) is required"
    launched = [node.value.func.value.id for node in ast.walk(launch)
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Subscript)
                and isinstance(node.value.func.value, ast.Name)]
    kernel = _function(tree, launched[0]) if len(launched) == 1 else None
    if kernel is None:
        return "launch must return one named JIT kernel"

    host_calls = [node for node in ast.walk(launch) if isinstance(node, ast.Call)]
    kernel_calls = [node for node in ast.walk(kernel) if isinstance(node, ast.Call)]
    host_named = lambda name: [call for call in host_calls if _call_name(call.func) == name]
    kernel_named = lambda name: [call for call in kernel_calls if _call_name(call.func) == name]
    issues: list[str] = []

    layouts = host_named("gl.NVMMASharedLayout.get_default_for")
    if len(layouts) != 3 or any(len(call.args) != 2 or call.keywords for call in layouts):
        issues.append("create three host shared layouts with get_default_for([block_m,block_k], gl.bfloat16) and two positional arguments")
    if kernel_named("gl.NVMMASharedLayout.get_default_for"):
        issues.append("reuse descriptor.layout inside the JIT kernel; do not call get_default_for there")
    descriptors = host_named("TensorDescriptor.from_tensor")
    if len(descriptors) != 3 or any(len(call.args) != 3 or call.keywords for call in descriptors):
        issues.append("use exactly a_desc=TensorDescriptor.from_tensor(A,[BM,BK],a_layout), b_desc=TensorDescriptor.from_tensor(B.T,[BN,BK],b_layout), c_desc=TensorDescriptor.from_tensor(C,[BM,BN],c_layout); no layout= keyword")
    if any(_call_name(node) == "triton.bfloat16" for node in ast.walk(launch)):
        issues.append("compare host tensor dtype with torch.bfloat16, not triton.bfloat16")

    shared = kernel_named("gl.allocate_shared_memory")
    if len(shared) != 3 or any(len(call.args) != 3 or call.keywords for call in shared):
        issues.append("allocate A/B/C with gl.allocate_shared_memory(descriptor.dtype, descriptor.block_shape, descriptor.layout)")
    if kernel_named("gl.allocate_shared"):
        issues.append("gl.allocate_shared is not the serial action shared-memory API")
    loads = kernel_named("tma.async_load")
    if len(loads) < 2 or any(
        len(call.args) != 4 or call.keywords
        or not isinstance(call.args[2], ast.Name) or "bar" not in call.args[2].id
        or not isinstance(call.args[3], ast.Name) or "smem" not in call.args[3].id
        for call in loads
    ):
        issues.append("use exactly tma.async_load(a_desc,[off_m,step*BK],bar,a_smem) and tma.async_load(b_desc,[off_n,step*BK],bar,b_smem); bar is the THIRD positional argument")
    load_descriptors = {
        call.args[0].id for call in loads
        if call.args and isinstance(call.args[0], ast.Name)
    }
    expects = kernel_named("mbarrier.expect")
    if not expects or any(
        len(call.args) != 2
        or not isinstance(call.args[1], ast.BinOp)
        or not isinstance(call.args[1].op, ast.Add)
        or any(not isinstance(operand, ast.Attribute)
               or operand.attr != "nbytes_per_cta"
               or not isinstance(operand.value, ast.Name)
               for operand in (call.args[1].left, call.args[1].right))
        or {call.args[1].left.value.id, call.args[1].right.value.id} != load_descriptors
        or len(load_descriptors) != 2
        for call in expects
    ):
        issues.append("set mbarrier.expect(bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta), or matching descriptor aliases used by tma.async_load; shared-memory .nbytes is not the TMA transfer count")
    if not kernel_named("c_smem.store") or kernel_named("gl.store"):
        issues.append("write BF16 output using c_smem.store(acc.to(gl.bfloat16)), not gl.store or shared indexing")
    if any(
        _call_name(call.func) == "gl.convert_layout"
        and any(keyword.arg == "layout"
                and isinstance(keyword.value, ast.Attribute)
                and keyword.value.attr == "layout"
                for keyword in call.keywords)
        for call in kernel_calls
    ):
        issues.append("do not convert_layout to descriptor.layout; write c_smem.store(acc.to(gl.bfloat16)) directly")
    if kernel_named("tma.store_commit"):
        issues.append("remove tma.store_commit; use tma.store_wait(pendings=0)")
    if kernel_named("mbarrier.inval") or not kernel_named("mbarrier.invalidate"):
        issues.append("end with mbarrier.invalidate(bar); mbarrier.inval does not exist")
    for loop in (node for node in ast.walk(kernel) if isinstance(node, (ast.For, ast.While))):
        if any(_call_name(call.func) == "warpgroup_mma" for statement in loop.body
               for call in ast.walk(statement) if isinstance(call, ast.Call)) and not any(
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == "acc"
            and isinstance(statement.value, ast.Call)
            and _call_name(statement.value.func) == "warpgroup_mma"
            for statement in loop.body
        ):
            issues.append("assign acc = warpgroup_mma(a_smem,b_smem.permute((1,0)),acc,is_async=True) in each K step")
        if any(isinstance(node, (ast.Assign, ast.AnnAssign))
               and isinstance(node.value, ast.Call)
               and _call_name(node.value.func) == "warpgroup_mma_wait"
               for statement in loop.body for node in ast.walk(statement)):
            issues.append("call warpgroup_mma_wait without assigning inside the loop; assign only after the loop")
            break
    return "; ".join(issues) if issues else None


def pipelined_wgmma_api_mismatch(parent_source: str, candidate_source: str) -> str | None:
    """Check the distinct three-slot wait-one structure before GPU evaluation.

    This is a source-shape guard only. Compilation, correctness, resource
    checks, and timing still decide whether the generated candidate is a node.
    """
    del parent_source
    try:
        tree = ast.parse(candidate_source)
    except SyntaxError:
        return "source is not valid Python"
    launch = _function(tree, "launch")
    if launch is None:
        return "launch(A,B,C,M,N,K) is required"
    launched = [node.value.func.value.id for node in ast.walk(launch)
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Subscript)
                and isinstance(node.value.func.value, ast.Name)]
    kernel = _function(tree, launched[0]) if len(launched) == 1 else None
    if kernel is None:
        return "launch must return one named JIT kernel"
    calls = [node for node in ast.walk(kernel) if isinstance(node, ast.Call)]
    named = lambda name: [call for call in calls if _call_name(call.func) == name]
    issues: list[str] = []
    shared = named("gl.allocate_shared_memory")
    if len(shared) < 3 or sum(
        bool(call.args and isinstance(call.args[1], ast.BinOp)
             and isinstance(call.args[1].left, ast.List)
             and len(call.args[1].left.elts) == 1
             and isinstance(call.args[1].left.elts[0], ast.Constant)
             and call.args[1].left.elts[0].value == 3)
        for call in shared if len(call.args) >= 2
    ) < 2:
        issues.append("allocate three-slot A and B shared storage using [3] + descriptor.block_shape")
    barriers = named("mbarrier.allocate_mbarrier")
    if len(barriers) != 1 or not any(
        keyword.arg == "batch" and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == 3
        for keyword in barriers[0].keywords
    ):
        issues.append("allocate exactly three mbarriers with batch=3")
    if len(named("mbarrier.init")) not in (1, 3) or len(named("mbarrier.invalidate")) not in (1, 3):
        issues.append("initialize and invalidate all three mbarriers, using a static loop or three explicit calls")
    if any(
        call.args and isinstance(call.args[0], (ast.List, ast.Tuple))
        for call in calls if _call_name(call.func) in {
            "a_smem.index", "b_smem.index", "bars.index",
        }
    ):
        issues.append("index shared tiles and barriers with scalar pre/slot, never [pre] or [slot]")
    if len(named("tma.async_load")) < 4:
        issues.append("issue A and B TMA loads in both the initial prefetch and guarded refill")
    expects = named("mbarrier.expect")
    if len(expects) < 2 or any(
        len(call.args) != 2 or {
            _call_name(node) for node in ast.walk(call.args[1])
            if isinstance(node, ast.Attribute) and node.attr == "nbytes_per_cta"
        } != {"a_desc.nbytes_per_cta", "b_desc.nbytes_per_cta"}
        for call in expects
    ):
        issues.append("use descriptor nbytes_per_cta for each prefetch/refill mbarrier.expect")
    if len(named("mbarrier.wait")) != 1:
        issues.append("wait for the current ring slot before its WGMMA issue")
    elif not any(keyword.arg == "phase" and "// 3" in ast.unparse(keyword.value)
                 for keyword in named("mbarrier.wait")[0].keywords):
        issues.append("use phase=(step // 3) % 2 for the reused ring slot")
    waits = named("warpgroup_mma_wait")
    outstanding = [keyword.value.value for call in waits for keyword in call.keywords
                   if keyword.arg == "num_outstanding"
                   and isinstance(keyword.value, ast.Constant)]
    if 1 not in outstanding or 0 not in outstanding:
        issues.append("use wait(num_outstanding=1) in the K loop and final wait(num_outstanding=0)")
    source = ast.unparse(kernel)
    if "step % 3" not in source or "step + 2" not in source or "future < steps" not in source:
        issues.append("use slot=step%3 and guarded future=step+2 refill")
    if any(keyword.arg == "multicast" and isinstance(keyword.value, ast.Constant)
           and keyword.value.value is True for call in named("tma.async_load")
           for keyword in call.keywords):
        issues.append("keep single-CTA B TMA loads unicast")
    return "; ".join(issues) if issues else None
