from src.dataset import get_dataloaders


def test_dataloader():

    train_loader, val_loader, test_loader = get_dataloaders()

    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0


def test_batch_shape():

    train_loader, _, _ = get_dataloaders()

    images, labels = next(iter(train_loader))

    assert images.shape[1:] == (3, 32, 32)
    assert len(images) == len(labels)