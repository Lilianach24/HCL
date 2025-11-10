import csv
import time

import torch

from model.loss import calculate_custom_loss, c_loss


class Trainer:
    def __init__(self, opt, model, train_loader, test_loader, optimizer, scheduler, device, logging, logs):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.dataset_name = opt.dataset_name
        self.num_epochs = opt.num_epochs
        self.logging = logging
        self.output_path = opt.output_path
        self.lambda_weight = opt.lambda_weight
        self.logs = logs
        # self.save_model_dir = logs['model']
        # self.save_log_dir = logs['logs']

    def valid(self):
        self.model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.test_loader):
                images = images.to(self.device).float()
                labels = labels.to(self.device)

                _, outputs = self.model(images)
                loss = c_loss(outputs, labels) 

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100. * test_correct / test_total

        return test_loss, test_acc

    def train(self):
        best_test_acc = 0.0

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (images, labels, s_values, label_probs) in enumerate(self.train_loader):
                images = images.to(self.device).float()
                labels = labels.to(self.device)
                s_values = s_values.to(self.device)
                label_probs = label_probs.to(self.device).float()

                self.optimizer.zero_grad()
                _, outputs = self.model(images)
                loss = calculate_custom_loss(outputs, labels, s_values, label_probs, self.lambda_weight)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

                if (batch_idx + 1) % 100 == 0:
                    self.logging.info(f'Epoch: {epoch + 1}/{self.num_epochs}, Batch: {batch_idx + 1}/{len(self.train_loader)}, '
                          f'Train Loss: {train_loss / (batch_idx + 1):.4f}, Train Acc: {100. * train_correct / train_total:.2f}%')

            # Learning rate adjustment
            if self.scheduler:
                self.scheduler.step()

            # test
            test_loss, test_acc = self.valid()

            # save best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                # self._save_model(epoch)

            self.logging.info(f'Epoch: {epoch + 1}/{self.num_epochs}, '
                  f'Train Loss: {train_loss / len(self.train_loader):.4f}, Train Acc: {100. * train_correct / train_total:.2f}%, '
                  f'Test Loss: {test_loss / len(self.test_loader):.4f}, Test Acc: {test_acc:.2f}%, learning rate: {self.optimizer.param_groups[0]["lr"]}')

            with open(self.logs, "a") as f:
                writer = csv.writer(f)
                if epoch == 0:
                    writer.writerow(["Epoch", "TrainAcc", "TestAcc", "TrainLoss", "TestLoss"])
                writer.writerow([epoch + 1, 100. * train_correct / train_total, test_acc, train_loss / len(self.train_loader), test_loss / len(self.test_loader)])
        #Save the final trained model
        # self._save_model()
        print(f'Best Test Accuracy: {best_test_acc:.2f}%')

    def _save_model(self):
        t = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        model_path = f'{self.output_path}/checkpoint_clip_{t}.pth'
        torch.save(self.model.state_dict(), model_path)

        print(f"The model has been saved to: {model_path}")
