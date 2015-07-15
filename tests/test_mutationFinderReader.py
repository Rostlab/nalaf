from unittest import TestCase

__author__ = 'carsten'


class TestMutationFinderReader(TestCase):

  @classmethod
  def setUpClass(cls):
      # create a sample dataset to test
      try:
        config.read_file(args.config)

        html_path = config.get('paths', 'html_path')
        ann_path = config.get('paths', 'ann_path')
        crf_path = config.get('paths', 'crf_path')

        # if only to check db then do that...
        if args.check_db:
            dbcheck.main(html_path=html_path, ann_path=ann_path)
            exit()

        dataset = VerspoorReader(html_path).read()

        if not args.quick_nl:
            NLTKSplitter().split(dataset)
            NLTKTokenizer().tokenize(dataset)

        VerspoorAnnotationReader(ann_path).annotate(dataset)

  def test_read(self):
    self.fail()
