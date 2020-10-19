from paddlehub.module.module import moduleinfo

@moduleinfo(
    name='test_module'
)
class TestModule:
    def __init__(self):
        print('This is a test module.')

    def echo(self, text):
        print(text)
