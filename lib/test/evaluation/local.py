from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/xyp/sx/SUTrack/data/got10k_lmdb'
    settings.got10k_path = '/home/xyp/sx/SUTrack/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = '/home/xyp/sx/SUTrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/xyp/sx/SUTrack/data/lasot_lmdb'
    settings.lasot_path = '/home/xyp/sx/SUTrack/data/lasot'
    settings.lasotlang_path = '/home/xyp/sx/SUTrack/data/lasot'
    settings.network_path = '/home/xyp/sx/SUTrack/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/xyp/sx/SUTrack/data/nfs'
    settings.otb_path = '/home/xyp/sx/SUTrack/data/OTB2015'
    settings.otblang_path = '/home/xyp/sx/SUTrack/data/otb_lang'
    settings.prj_dir = '/home/xyp/sx/SUTrack'
    settings.result_plot_path = '/home/xyp/sx/SUTrack/test/result_plots'
    settings.results_path = '/home/xyp/sx/SUTrack/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/xyp/sx/SUTrack'
    settings.segmentation_path = '/home/xyp/sx/SUTrack/test/segmentation_results'
    settings.tc128_path = '/home/xyp/sx/SUTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/xyp/sx/SUTrack/data/tnl2k/test'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/xyp/sx/SUTrack/data/trackingnet'
    #settings.uav_path = '/home/xyp/sx/SUTrack/data/UAVTest'
    # 新增
    settings.uav_predict_path = '/home/xyp/sx/SUTrack/data/UAV_predict'
    settings.uav_test_path = '/home/xyp/sx/SUTrack/data/UAVTest'
    settings.vot_path = '/home/xyp/sx/SUTrack/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

