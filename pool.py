from concurrent.futures import ProcessPoolExecutor as Pool

def main():
    nproc = 55
    with Pool(nproc) as pool:
        for i, output_data in enumerate(pool.map(calc_all_observables, params)):


if __name__ == '__main__':
    main()