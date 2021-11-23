import aiomas
import arrow
import click
from mat4py import loadmat
import numpy as np


def get_container_kwargs(start_date):
	"""Return a dictionary with keyword arguments *(kwargs)* used by both, the
	root container, and the containers in the sub processes.
	*start_date* is an Arrow date-time object used to initialize the container
	clock.
	"""
	return {
		'clock': aiomas.ExternalClock(start_date, init_time=0),
		'codec': aiomas.MsgPackBlosc,
		'extra_serializers':[get_np_serializer],
	}


def validate_addr(ctx, param, value):

	"""*Click* validator that makes sure that *value* is a valid address
	*host:port*."""
	try:
		host, port = value.rsplit(':', 1)
		return (host, int(port))
	except ValueError as e:
		raise click.BadParameter(e)


def validate_start_date(ctx, param, value):
	"""*Click* validator that makes sure that *value* is a date string that
	*arrow* can parse."""
	try:
		arrow.get(value)  # Check if the date can be parsed
	except arrow.parser.ParserError as e:
		raise click.BadParameter(e)
	return value

def read_data(path):
	''' this function is adapted to the data format given must be checked for flexibility'''
	d = loadmat(path)
	data = {}
	for key, dt in d.items():
		if not isinstance(dt, list):# here read integer such as BCT
			dt-=1 # from matlab adaptation
			data[key] = np.array([dt])

		else: #here read list and matrices
			if key == 'UserNode':
				dt = [x - 1 for x in dt] #from matlab adaptation
			data[key] = np.array(dt)

	return data

def get_np_serializer():
   """Return a tuple *(type, serialize(), deserialize())* for NumPy arrays
   for usage with an :class:`aiomas.codecs.MsgPack` codec.

   """
   return np.ndarray, _serialize_ndarray, _deserialize_ndarray


def _serialize_ndarray(obj):
   return {
      'type': obj.dtype.str,
      'shape': obj.shape,
      'data': obj.tostring(),
	   #'data': obj.tobytes(),
   }


def _deserialize_ndarray(obj):
   array = np.fromstring(obj['data'], dtype=np.dtype(obj['type']))
   #array = np.frombuffer(obj['data'], dtype=np.dtype(obj['type']))
   return array.reshape(obj['shape'])
