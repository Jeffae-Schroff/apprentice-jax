import h5py
import numpy as np

def merge_sys_errors(input_h5, output_path, include_stat_error, include_pdf_error=True, include_isr_error=True, include_scale_error=True):
    f = h5py.File(input_h5, "r")

    #We start with the nominal values listed together with systematic errors in one big column under the key "values"
    values = np.array(f['values'][:], dtype=np.float64) #num_bins * num_mc_runs array
    num_mc_runs = np.shape(values)[1]
    stat_err = np.array(f['errors'][:], dtype=np.float64)
    bin_ids = np.array([x.decode() for x in f.get("index")[:]])
    
    id_cut = np.array([sb[0] for sb in np.char.rsplit(bin_ids, "#", maxsplit=1)])
    num_cut = np.array([sb[1] for sb in np.char.rsplit(bin_ids, "#", maxsplit=1)])

    is_obs = np.array([(not "[" in str) and (not "]" in str) for str in id_cut])
    if include_isr_error:
        is_isr_lower = np.array(["[AUX_isr:pdf:minus=1.0]" in str for str in id_cut])
        is_isr_upper = np.array(["[AUX_isr:pdf:plus=1.0]" in str for str in id_cut])
    if include_pdf_error:
        is_pdf_lower = np.array(["[AUX_pdfdn]" in str for str in id_cut])
        is_pdf_upper = np.array(["[AUX_pdfup]" in str for str in id_cut])
    if include_scale_error:
        is_scale_lower = np.array(["[AUX_mur0.5_muf1]" in str for str in id_cut])
        is_scale_upper = np.array(["[AUX_mur2_muf_1]" in str for str in id_cut])

    id_cut_cut = np.array([sb[0] for sb in np.char.rsplit(id_cut, "[", maxsplit=1)])

    obs_index = id_cut[is_obs]
    
    nominal_values = values[is_obs]
    if include_stat_error:
        nomval_stat_err = stat_err[is_obs]

    

    obs_names, obs_counts = np.unique(obs_index, return_counts=True)
    
    total_err = np.array([])
    
    for i, obs_name in enumerate(obs_names):
        obs_bin_count = obs_counts[i]
        obs_values = values[id_cut_cut == obs_name]
        obs_nom_val = nominal_values[obs_index == obs_name]
        obs_stat_err = nomval_stat_err[obs_index == obs_name]
        obs_err_sq = np.zeros((obs_bin_count, num_mc_runs))
        if include_isr_error:
            is_isr_lower_obs = is_isr_lower[id_cut_cut == obs_name]
            is_isr_upper_obs = is_isr_upper[id_cut_cut == obs_name]
            if np.sum(is_isr_lower_obs)==obs_bin_count and np.sum(is_isr_upper_obs)==obs_bin_count:
                isr_lower_obs = obs_values[is_isr_lower_obs]
                isr_upper_obs = obs_values[is_isr_upper_obs]
                isr_sq = np.square(np.maximum(obs_nom_val - isr_lower_obs, isr_upper_obs - obs_nom_val))
                obs_err_sq = obs_err_sq + isr_sq
            else:
                print("Observable ", obs_name, " missing isr systematic error bins.")
        if include_scale_error:
            is_scale_lower_obs = is_scale_lower[id_cut_cut == obs_name]
            is_scale_upper_obs = is_scale_upper[id_cut_cut == obs_name]
            if np.sum(is_scale_lower_obs)==obs_bin_count and np.sum(is_scale_upper_obs)==obs_bin_count:
                scale_lower_obs = obs_values[is_scale_lower_obs]
                scale_upper_obs = obs_values[is_scale_upper_obs]
                scale_sq = np.square(np.maximum(obs_nom_val - scale_lower_obs, scale_upper_obs - obs_nom_val))
                obs_err_sq = obs_err_sq + scale_sq
            else:
                print("Observable ", obs_name, " missing scale systematic error bins.")
        if include_pdf_error:
            is_pdf_lower_obs = is_pdf_lower[id_cut_cut == obs_name]
            is_pdf_upper_obs = is_pdf_upper[id_cut_cut == obs_name]
            if np.sum(is_pdf_lower_obs)==obs_bin_count and np.sum(is_scale_upper_obs)==obs_bin_count:
                pdf_lower_obs = obs_values[is_pdf_lower_obs]
                pdf_upper_obs = obs_values[is_pdf_upper_obs]
                pdf_sq = np.square(np.average(obs_nom_val - pdf_lower_obs)) + np.square(np.average(pdf_upper_obs - obs_nom_val))
                obs_err_sq = obs_err_sq + pdf_sq
            else:
                print("Observable ", obs_name, " missing pdf systematic error bins.")
        if include_stat_error:
            obs_err_sq = obs_err_sq + np.square(obs_stat_err)
        if i==0:
            total_err = np.sqrt(obs_err_sq)
        else:
            total_err = np.concatenate((total_err, np.sqrt(obs_err_sq)), axis=0)
        

    f2 = h5py.File(output_path, 'w')
    f.copy('params', f2)
    f.copy('runs', f2)
    f.copy('xmax', f2)
    f.copy('xmin', f2)
    f2.create_dataset('values', data=nominal_values)
    f2.create_dataset('index', data=[a.encode('ascii', 'ignore') for a in bin_ids[is_obs]])
    f2.create_dataset('total_err', data=total_err)
    f2.close()


    if include_stat_error:
        print("Total error combined into new column, written to ", output_path)
    else:
        print("Systematic errors combined into new column, written to ", output_path)