__all__ = [
	'assert_fails'
]

def assert_fails(fun, expected=ValueError, *args, **kwargs):
	failed = False
	try:
		fun(*args, **kwargs)
	except expected as e:
		failed = True
	assert failed, 'expected %s in function' % expected
