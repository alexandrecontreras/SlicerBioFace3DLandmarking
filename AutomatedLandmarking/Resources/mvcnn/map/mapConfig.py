# Config key mapping
global name, ngpu, arch, pd, dl, mdl, opt, loss, met, sched, train, pal, pred
name = "name"
ngpu = "n_gpu"
arch = "arch"
pd = "preparedata"
dl = "data_loader"
mdl = "process_3d"
opt = "optimizer"
loss = "loss"
met = "metrics"
sched = "lr_scheduler"
train = "trainer"
pal = "pre-align"
pred = "predict"

# Architecture key mapping
global arch_type, arch_args
arch_type = "type"
arch_args = "args"
global arch_args_nlm, arch_args_nfeat, arch_args_droprate, arch_args_imgch
arch_args_nlm = "n_landmarks"
arch_args_nfeat = "n_features"
arch_args_droprate = "dropout_rate"
arch_args_imgch = "image_channels"

# PrepareData key mapping
global pd_rawJson, pd_processedDir, pd_rendering, pd_trainf, pd_testf, pd_split
pd_rawJson = "raw_data_json"
pd_processedDir = "processed_data_dir"
pd_splitDir = "split_data_dir"
pd_trainf = "train_file"
pd_valf = "val_file"
pd_testf = "test_file"
pd_dumpf = "dump_file"
pd_split = "split"

# DataLoader mapping
global dl_type, dl_args
dl_type = "type"
dl_args = "args"
global dl_args_datadir, dl_args_hmapsize, dl_args_imgch, dl_args_batchsize, dl_args_shuffle
global dl_args_valsplit, dl_args_workers, dl_args_imgsize, dl_args_rendering, dl_args_views
dl_args_datadir = "data_dir"
dl_args_hmapsize = "heatmap_size"
dl_args_imgch = "image_channels"
dl_args_batchsize = "batch_size"
dl_args_shuffle = "shuffle"
dl_args_valsplit = "validation_split"
dl_args_workers = "num_workers"
dl_args_imgsize = "image_size"
dl_args_views = "n_views"

# Model key mapping
global mdl_minx, mdl_maxx, mdl_miny, mdl_maxy, mdl_minz, mdl_maxz, mdl_offrender, mdl_vlines, mdl_maxq, mdl_thresh, mdl_render
mdl_minx = "min_x_angle"
mdl_maxx = "max_x_angle"
mdl_miny = "min_y_angle"
mdl_maxy = "max_y_angle"
mdl_minz = "min_z_angle"
mdl_maxz = "max_z_angle"
mdl_offrender = "off_screen_rendering"
mdl_vlines = "filter_view_lines"
mdl_maxq = "heatmap_max_quantile"
mdl_thresh = "heatmap_abs_threshold"
mdl_render = "write_renderings"

# Optimizer key mapping
global opt_type, opt_args
opt_type = "type"
opt_args = "args"
global opt_args_lr, opt_args_wdecay, opt_args_amsgrad
opt_args_lr = "lr"
opt_args_wdecay = "weight_decay"
opt_args_amsgrad = "amsgrad"

# Scheduler key mapping
global sched_type, sched_args
sched_type = "type"
sched_args = "args"
global sched_args_step, sched_args_gamma
sched_args_step = "step_size"
sched_args_gamma = "gamma"

# Trainer key mapping
global train_savedir, train_epochs, train_save, train_verb, train_monitor, train_stop, train_tensor
train_savedir = "save_dir"
train_epochs = "epochs"
train_save = "save_period"
train_verb = "verbosity"
train_monitor = "monitor"
train_stop = "early_stop"
train_tensor = "tensorboard"
train_keep = "keepEpochs"

# Pre-align key mapping
global pal_com, pal_rx, pal_ry, pal_rz, pal_scale, pal_write
pal_com = "align_center_of_mass"
pal_rx = "rot_x"
pal_ry = "rot_y"
pal_rz = "rot_z"
pal_scale = "scale"
pal_write = "write_pre_aligned"

# Predict key mapping
global pred_mdl
pred_mdl = "model_pth_or_url"
