from src.data.__local__ import implemented_datasets


def load_dataset(learner, dataset_name, pretrain=False):

    assert dataset_name in implemented_datasets

    if dataset_name == "mnist":
        from src.data.mnist import MNIST_DataLoader
        print("from main for mnist")
        data_loader = MNIST_DataLoader
        # load data with data loader
        print("main into learner.load_data")
        learner.load_data(data_loader=data_loader, pretrain=pretrain)
        print("Out learner.load_data ")
        # check all parameters have been attributed
        learner.data.check_all()
        print("Out learner.check_all ")

    if dataset_name == "cifar10":
        from src.data.cifar10 import CIFAR_10_DataLoader
        print("from main for cifar10")
        data_loader = CIFAR_10_DataLoader
        # load data with data loader
        print("main into learner.load_data")
        learner.load_data(data_loader=data_loader, pretrain=pretrain)
        print("Out learner.load_data ")
        # check all parameters have been attributed
        learner.data.check_all()
        print("Out learner.check_all ")

    if dataset_name == "cifar10prnt":
        from src.data.cifar10Prnt import CIFAR_10_DataLoader
        print("from main for cifar10Prnt")
        data_loader = CIFAR_10_DataLoader
        # load data with data loader
        print("main into learner.load_data")
        learner.load_data(data_loader=data_loader, pretrain=pretrain)
        print("Out learner.load_data ")
        # check all parameters have been attributed
        learner.data.check_all()
        print("Out learner.check_all ")

    if dataset_name == "cifar10nw":
        print("from main into cifar10NW.py with cifar10nw")
        from src.data.cifar10NW import CIFAR_10_DataLoader
        print("from main for cifar10nw")
        data_loader = CIFAR_10_DataLoader
        # load data with data loader
        print("(cifnw) main into learner.load_data")
        learner.load_data(data_loader=data_loader, pretrain=pretrain)
        print("(cifnw) Out learner.load_data ")
        # check all parameters have been attributed
        learner.data.check_all()
        print("(cifnw) Out learner.check_all ")

    if dataset_name == "gtsrb":
        from src.data.GTSRB import GTSRB_DataLoader
        print("from main for gtsrb")
        data_loader = GTSRB_DataLoader
        # load data with data loader
        print("main into learner.load_data")
        learner.load_data(data_loader=data_loader, pretrain=pretrain)
        print("Out learner.load_data ")
        # check all parameters have been attributed
        learner.data.check_all()
        print("Out learner.check_all ")

    if dataset_name == "dogs":
        from src.data.dDOGS import DOGS_DataLoader
        print("from main for dogs")
        data_loader = DOGS_DataLoader
        # load data with data loader
        print("main into learner.load_data")
        learner.load_data(data_loader=data_loader, pretrain=pretrain)
        print("Out learner.load_data ")
        # check all parameters have been attributed
        learner.data.check_all()
        print("Out learner.check_all ")

    if dataset_name == "catsdogs":
        from src.data.cDOGS import cDOGS_DataLoader
        print("from main for catsdogs")
        data_loader = cDOGS_DataLoader
        # load data with data loader
        print("main into learner.load_data")
        learner.load_data(data_loader=data_loader, pretrain=pretrain)
        print("Out learner.load_data ")
        # check all parameters have been attributed
        learner.data.check_all()
        print("Out learner.check_all ")

    if dataset_name == "adi":
        from src.data.ADI import ADI_DataLoader
        print("from main for adi")
        data_loader = ADI_DataLoader
        # load data with data loader
        print("main into learner.load_data")
        learner.load_data(data_loader=data_loader, pretrain=pretrain)
        print("Out learner.load_data ")
        # check all parameters have been attributed
        learner.data.check_all()
        print("Out learner.check_all ")


