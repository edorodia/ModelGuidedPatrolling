mkdir "ModelTrain/NoNoise_Shekel_Datasets"

::NoNoise	-	True	True	miopic
python .\ModelTrain\static_dataset_creation-2_in_het.py  --drone_noise NoNoise   --influence_drone_visited_map   --influence_asv_visited_map     --importance_asv_read miopic
set "folderName=NoNoise_-_True_True_miopic"
mkdir "ModelTrain/NoNoise_Shekel_Datasets/%folderName%"
copy ModelTrain\Data\gts_shekel_train.npy ModelTrain\NoNoise_Shekel_Datasets\%folderName%\
copy ModelTrain\Data\trajectories_shekel_train.npy ModelTrain\NoNoise_Shekel_Datasets\%folderName%\

::NoNoise	-	True	True	none
python .\ModelTrain\static_dataset_creation-2_in_het.py  --drone_noise NoNoise   --influence_drone_visited_map   --influence_asv_visited_map     --importance_asv_read none
set "folderName=NoNoise_-_True_True_none"
mkdir "ModelTrain/NoNoise_Shekel_Datasets/%folderName%"
copy ModelTrain\Data\gts_shekel_train.npy ModelTrain\NoNoise_Shekel_Datasets\%folderName%\
copy ModelTrain\Data\trajectories_shekel_train.npy ModelTrain\NoNoise_Shekel_Datasets\%folderName%\

::NoNoise	-	False	False	miopic
python .\ModelTrain\static_dataset_creation-2_in_het.py  --drone_noise NoNoise                                                                   --importance_asv_read miopic
set "folderName=NoNoise_-_False_False_miopic"
mkdir "ModelTrain/NoNoise_Shekel_Datasets/%folderName%"
copy ModelTrain\Data\gts_shekel_train.npy ModelTrain\NoNoise_Shekel_Datasets\%folderName%\
copy ModelTrain\Data\trajectories_shekel_train.npy ModelTrain\NoNoise_Shekel_Datasets\%folderName%\

::NoNoise	-	False	False	none
python .\ModelTrain\static_dataset_creation-2_in_het.py  --drone_noise NoNoise                                                                   --importance_asv_read none
set "folderName=NoNoise_-_False_False_none"
mkdir "ModelTrain/NoNoise_Shekel_Datasets/%folderName%"
copy ModelTrain\Data\gts_shekel_train.npy ModelTrain\NoNoise_Shekel_Datasets\%folderName%\
copy ModelTrain\Data\trajectories_shekel_train.npy ModelTrain\NoNoise_Shekel_Datasets\%folderName%\

::NoNoise	-	True	False	miopic
python .\ModelTrain\static_dataset_creation-2_in_het.py  --drone_noise NoNoise   --influence_drone_visited_map                                   --importance_asv_read miopic
set "folderName=NoNoise_-_True_False_miopic"
mkdir "ModelTrain/NoNoise_Shekel_Datasets/%folderName%"
copy ModelTrain\Data\gts_shekel_train.npy ModelTrain\NoNoise_Shekel_Datasets\%folderName%\
copy ModelTrain\Data\trajectories_shekel_train.npy ModelTrain\NoNoise_Shekel_Datasets\%folderName%\

::NoNoise	-	True	False	none
python .\ModelTrain\static_dataset_creation-2_in_het.py  --drone_noise NoNoise   --influence_drone_visited_map                                   --importance_asv_read none
set "folderName=NoNoise_-_True_False_none"
mkdir "ModelTrain/NoNoise_Shekel_Datasets/%folderName%"
copy ModelTrain\Data\gts_shekel_train.npy ModelTrain\NoNoise_Shekel_Datasets\%folderName%\
copy ModelTrain\Data\trajectories_shekel_train.npy ModelTrain\NoNoise_Shekel_Datasets\%folderName%\

::NoNoise	-	False	True	miopic
python .\ModelTrain\static_dataset_creation-2_in_het.py  --drone_noise NoNoise                                   --influence_asv_visited_map     --importance_asv_read miopic
set "folderName=NoNoise_-_False_True_miopic"
mkdir "ModelTrain/NoNoise_Shekel_Datasets/%folderName%"
copy ModelTrain\Data\gts_shekel_train.npy ModelTrain\NoNoise_Shekel_Datasets\%folderName%\
copy ModelTrain\Data\trajectories_shekel_train.npy ModelTrain\NoNoise_Shekel_Datasets\%folderName%\

::NoNoise	-	False	True	none
python .\ModelTrain\static_dataset_creation-2_in_het.py  --drone_noise NoNoise                                   --influence_asv_visited_map     --importance_asv_read none
set "folderName=NoNoise_-_False_True_none"
mkdir "ModelTrain/NoNoise_Shekel_Datasets/%folderName%"
copy ModelTrain\Data\gts_shekel_train.npy ModelTrain\NoNoise_Shekel_Datasets\%folderName%\
copy ModelTrain\Data\trajectories_shekel_train.npy ModelTrain\NoNoise_Shekel_Datasets\%folderName%\
