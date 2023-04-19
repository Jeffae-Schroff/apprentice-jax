import argparse
import apprentice as app
from mpi4py import MPI

INVALID_NUMBER = -999999.


def readInputDataYODA(dirnames, parFileName="params.dat", wfile=None, storeAsH5=None, comm = MPI.COMM_WORLD):
    import apprentice as app
    import numpy as np
    import yoda, glob, os
    size = comm.Get_size()
    rank = comm.Get_rank()

    indirs=None
    dirnames = [dirnames]
    if rank==0:
        INDIRSLIST = [glob.glob(os.path.join(a, "*")) for a in dirnames]
        indirs     = [item for sublist in INDIRSLIST for item in sublist]
    indirs = comm.bcast(indirs, root=0)


    rankDirs = app.tools.chunkIt(indirs, size) if rank==0 else None
    rankDirs = comm.scatter(rankDirs, root=0)

    PARAMS, HISTOS = app.io.read_rundata(rankDirs, parFileName)
    send = []
    for k, v in HISTOS.items():
        temp = []
        for _k, _v in v.items():
            temp.append((_k, _v))
        send.append((k, temp))

    params = comm.gather(PARAMS, root=0)
    histos = comm.gather(send, root=0)

    rankIdx, binids, X, Y, E, xmin, xmax, pnames, BNAMES, data= None, None, None, None, None, None, None, None, None, None
    if rank==0:
        _params = {}
        _histos = {}
        for p in params: _params.update(p)

        for rl in histos:
            for ih in range(len(rl)):
                hname = rl[ih][0]
                if not hname in _histos: _histos[hname] = {}
                for ir in range(len(rl[ih][1])):
                    run =  rl[ih][1][ir][0]
                    _histos[hname][run] = rl[ih][1][ir][1]

        pnames = [str(x) for x in _params[list(_params.keys())[0]].keys()]
        runs = sorted(list(_params.keys()))
        X=np.array([list(_params[r].values()) for r in runs])
        # Iterate through all histos, bins and mc runs to rearrange data
        hbins ={}
        HNAMES=[str(x) for x in sorted(list(_histos.keys()))]
        if wfile is not None:
            observables = list(set(app.io.readObs(wfile)))
            HNAMES = [hn for hn in HNAMES if hn in observables]
        BNAMES = []
        for hn in HNAMES:
            histos = _histos[hn]
            nbins = len(list(histos.values())[0])
            hbins[hn]=nbins
            for n in range(nbins):
                BNAMES.append("%s#%i"%(hn, n))

        _data, xmin, xmax = [], [], []
        for hn in HNAMES:
            for nb in range(hbins[hn]):
                vals = np.array([_histos[hn][r][nb][2] if r in _histos[hn].keys() else np.nan for r in runs])
                errs = np.array([_histos[hn][r][nb][3] if r in _histos[hn].keys() else np.nan for r in runs])
                # Pick a run that actually exists here
                goodrun = runs[np.where(np.isfinite(vals))[0][0]]
                xmin.append(_histos[hn][goodrun][nb][0])
                xmax.append(_histos[hn][goodrun][nb][1])
                USE = (~np.isinf(vals)) & (~np.isnan(vals)) & (~np.isinf(errs)) & (~np.isnan(errs))
                xg = X 
                if len(xg.shape)==3:
                    xg=xg.reshape(xg.shape[1:])
                vals[~USE] = INVALID_NUMBER
                errs[~USE] = INVALID_NUMBER
                _data.append([xg, np.array(vals), np.array(errs)])

        if storeAsH5 is not None:
            writeInputDataSetH5(storeAsH5, _data, runs, BNAMES, pnames, xmin, xmax)

        # TODO add weight file reading for obsevable filtering
        observables = np.unique([x.split("#")[0] for x in BNAMES])
        im   = {ls: np.where(np.char.find(BNAMES, ls) > -1)[0] for ls in observables}
        IDX  = np.sort(np.concatenate(list(im.values())))

        rankIdx = app.tools.chunkIt(IDX, size)
        data = app.tools.chunkIt(_data, size)
        xmin = app.tools.chunkIt(xmin, size)
        xmax = app.tools.chunkIt(xmax, size)
        binids = app.tools.chunkIt(BNAMES, size)
    rankIdx = comm.scatter(rankIdx, root=0)
    data = comm.scatter(data, root=0)
    xmin = comm.scatter(xmin, root=0)
    xmax = comm.scatter(xmax, root=0)
    binids = comm.scatter(binids, root=0)
    pnames = comm.bcast(pnames, root=0)


    comm.barrier()

    return data, binids, pnames, rankIdx, xmin, xmax

def writeInputDataSetH5(fname, data, runs, BNAMES, pnames, xmin, xmax, compression=4):
    import h5py
    import numpy as np
    f = h5py.File(fname, "w")

    # TODO change encoding to fixed size ascii
    # https://github.com/h5py/h5py/issues/892
    f.create_dataset("runs",  data=np.char.encode(runs,   encoding='utf8'),  compression=compression)
    f.create_dataset("index", data=np.char.encode(BNAMES, encoding='utf8'),  compression=compression)
    pset = f.create_dataset("params", data=data[0][0], compression=compression)
    pset.attrs["names"] = [x.encode('utf8') for x in pnames]

    f.create_dataset("values", data=np.array([d[1] for d in data]), compression=compression)
    f.create_dataset("errors", data=np.array([d[2] for d in data]), compression=compression)
    f.create_dataset("xmin", data=xmin, compression=compression)
    f.create_dataset("xmax", data=xmax, compression=compression)
    f.close()

def main():
    parser = argparse.ArgumentParser(description = 'Convert yoda file to HDF5')
    parser.add_argument(
        'filepath',
        help = 'the folder where we are making the h5 file')
    args = parser.parse_args()
    readInputDataYODA(args.filepath, parFileName="used_parameters", storeAsH5="zmumu.h5")

if __name__ == "__main__":
    main()