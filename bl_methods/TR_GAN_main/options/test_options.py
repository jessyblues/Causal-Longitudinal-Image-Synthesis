from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        #parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='inference', help='train, test, inference etc')
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        # Dropout and Batchnorm has different behavioir during training and test.
        #parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        #parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        # rewrite devalue values
        #parser.set_defaults(model='test')
        self.isTrain = False
        return parser
