if __name__ == "__main__":
    import v6_STFT_analysis as STFT
    from multiprocessing import Pool
    import time

    ionfolders = STFT.choose_top_folders(".data")
    print(ionfolders)
    file_ending = ".A.bin"
    filelists = STFT.generate_filelist(ionfolders, file_ending)
    savedir = STFT.choose_save_folder()

    for i, (filelist, folder) in enumerate(zip(filelists, ionfolders)):
        t0 = time.time()
        p = Pool(8)
        # STFT.one_file is function defined in STFT_analysis
        zipStruct = zip(filelist, [savedir] * len(filelist))
        all_results = p.starmap(STFT.one_file, zipStruct)
        p.close()
        p.join()
        t1 = time.time()
        total = t1 - t0
        print("FFT time is {}".format(total))

