from tkinter import * 


class Main(object):
	"""Deep Learning Main Menu"""
	def __init__(self, arg):
		super(Main, self).__init__()
		self.arg = arg
		self.window = Tk()
		self.window.title(arg)
		self.window.mainloop()
