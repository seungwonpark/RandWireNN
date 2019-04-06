from tensorboardX import SummaryWriter


class MyWriter(SummaryWriter):
    def __init__(self, logdir):
        super(MyWriter, self).__init__(logdir)

    def log_training(self, train_loss, step):
        self.add_scalar('loss/train_loss', train_loss, step)

    def log_evaluation(self, test_loss, accuracy, step):
        self.add_scalar('loss/test_loss', test_loss, step)
        self.add_scalar('eval/Top1_accuracy', accuracy, step)
