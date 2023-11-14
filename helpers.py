from solve import Solve
from polynomial_builder import PolynomialBuilder
import itertools
from concurrent import futures
from stqdm import stqdm

def get_result(params, pbar_container, max_deg=15):
    x1_range = list(range(1, max_deg+1)) if params['degrees'][0] == 0 else [params['degrees'][0]]
    x2_range = list(range(1, max_deg+1)) if params['degrees'][1] == 0 else [params['degrees'][1]]
    x3_range = list(range(1, max_deg+1)) if params['degrees'][2] == 0 else [params['degrees'][2]]

    ranges = list(itertools.product(x1_range, x2_range, x3_range, [params]))
    if len(ranges) > 1:
        with futures.ThreadPoolExecutor() as pool:
            results = list(stqdm(
                pool.map(get_error, ranges), 
                total=len(ranges), 
                st_container=pbar_container,
                desc='підбір',
                backend=True, frontend=False))
        results.sort(key=lambda t: t[1])
    else:
        results = [get_error(ranges[0])]

    final_params = params.copy()
    final_params['degrees'] = results[0][0]
    solver = Solve(final_params)
    solver.prepare()
    solution = PolynomialBuilder(solver)
    
    return solver, solution, final_params['degrees']

def get_error(params):
    params_new = params[-1].copy()
    params_new['degrees'] = [*(params[:-1])]
    solver = Solve(params_new)
    func_runtimes = solver.prepare()
    normed_error = min(solver.norm_error)
    return (params_new['degrees'], normed_error, func_runtimes)