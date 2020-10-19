from paddlehub.module.module import moduleinfo

@moduleinfo(
    name='test_module',
    version='1.0.0'
)
class TestModule:
    def __init__(self):
        print('This is a test module.')

    def echo(self, text):
        print(text)
