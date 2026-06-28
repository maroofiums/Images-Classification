from tqdm import tqdm
import torch

from src.metrics import calculate_metrics

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):
            

    model.train()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    progress_bar =  tqdm(
        dataloader,
        desc="Training...",
        leave=False
    )

    for images, labels in dataloader:
        
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        predictions = torch.argmax(outputs,dim = 1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )
        all_labels.extend(
            labels.cpu().numpy()
        )

        progress_bar.set_postfix(
            loss=f"{loss.items():.4f}"
        )

    avg_loss = running_loss / len(dataloader)

    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )

    return avg_loss, metrics

def validation_one_epoch(
    model,
    dataloader,
    criterion,
    device
):
    model.eval()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        progress_bar =  tqdm(
            dataloader,
            desc="Validation...",
            leave=False
        )

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)
                    
            outputs = model(images)

            loss = criterion(outputs, labels)


            running_loss += loss.item()
            predictions = torch.argmax(outputs,dim = 1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )
            all_labels.extend(
                labels.cpu().numpy()
            )

            progress_bar.set_postfix(
                loss=f"{loss.items():.4f}"
            )

    avg_loss = running_loss / len(dataloader)

    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )

    return avg_loss, metrics