
import torch
import os

class MovingAverage():
	""" Keeps an average window of the specified number of items. """

	def __init__(self, max_window_size=1000):
		self.max_window_size = max_window_size
		self.window = []
		self.sum = 0

	def add(self, elem):
		""" Adds an element to the window, removing the earliest element if necessary. """
		self.window.append(elem)
		self.sum += elem

		if len(self.window) > self.max_window_size:
			self.sum -= self.window.pop(0)
	
	def append(self, elem):
		""" Same as add just more pythonic. """
		self.add(elem)

	def get_avg(self):
		""" Returns the average of the elements in the window. """
		return self.sum / max(len(self.window), 1)

	def __str__(self):
		return str(self.get_avg())
	
	def __repr__(self):
		return repr(self.get_avg())


class ProgressBar():
	""" A simple progress bar that just outputs a string. """

	def __init__(self, length, max_val):
		self.max_val = max_val
		self.length = length
		self.cur_val = 0
		
		self.cur_num_bars = -1
		self._update_str()

	def set_val(self, new_val):
		self.cur_val = new_val

		if self.cur_val > self.max_val:
			self.cur_val = self.max_val
		if self.cur_val < 0:
			self.cur_val = 0

		self._update_str()
	
	def is_finished(self):
		return self.cur_val == self.max_val

	def _update_str(self):
		num_bars = int(self.length * (self.cur_val / self.max_val))

		if num_bars != self.cur_num_bars:
			self.cur_num_bars = num_bars
			self.string = '█' * num_bars + '░' * (self.length - num_bars)
	
	def __repr__(self):
		return self.string
	
	def __str__(self):
		return self.string

def sanitize_coordinates(_x1, _x2, img_size, cast=True):
	"""
	Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
	Also converts from relative to absolute coordinates and casts the results to long tensors.

	If cast is false, the result won't be cast to longs.
	"""
	_x1 *= img_size
	_x2 *= img_size
	if cast:
		_x1 = _x1.long()
		_x2 = _x2.long()
	x1 = torch.min(_x1, _x2)
	x2 = torch.max(_x1, _x2)
	x1 = torch.clamp(x1-1, min=0)
	x2 = torch.clamp(x2+1, max=img_size)

	return x1, x2


def init_console():
	"""
	Initialize the console to be able to use ANSI escape characters on Windows.
	"""
	if os.name == 'nt':
		from colorama import init
		init()
