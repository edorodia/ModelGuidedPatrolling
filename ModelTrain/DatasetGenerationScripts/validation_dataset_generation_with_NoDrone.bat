mkdir "ModelTrain/validation_NoDrone_Shekel_Datasets"

::NoDrone	-	False	none ##############################
start cmd /k ^
    title Command_1                                                                                                                                                             ^& ^
    mkdir "ModelTrain/validation_NoDrone_Shekel_Datasets/NoDrone_-_False_none"                                                                                            ^& ^
    python .\ModelTrain\static_dataset_creation.py  --N_episodes 100    --set validation    ^& ^
    copy ModelTrain\Data\gts_shekel_validation.npy ModelTrain\validation_NoDrone_Shekel_Datasets\NoDrone_-_False_none\gts_shekel_validation.npy                               ^& ^
    copy ModelTrain\Data\trajectories_shekel_validation.npy ModelTrain\validation_NoDrone_Shekel_Datasets\NoDrone_-_False_none\trajectories_shekel_validation.npy
