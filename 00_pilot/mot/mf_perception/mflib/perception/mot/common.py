import pickle
class FrameGetter:
  def __init__(self):
    self.cur_fname = ''
    self.frames = None
    pass

  def get_frame(self, frame_fname, frame_idx):
    if self.cur_fname != frame_fname:
      self.cur_fname = frame_fname
      self.frames = pickle.load(open(frame_fname, 'rb'))
    assert frame_idx < len(self.frames)
    frame = self.frames[frame_idx]
    return frame

class FrameMOT3D:
  def __init__(self, *args): # rgb, depth, detections, odom
    if len(args) == 4:
      self.rgb        = args[0]
      self.depth      = args[1]
      self.detections = args[2]
      self.odom       = args[3]
      self.ts         = self.detections.ts
    elif len(args) == 0:
      self.rgb        = None
      self.depth      = None
      self.detections = None
      self.odom       = None
      self.ts         = None
    else:
      raise ValueError('invalid number of arguments')
  def print_ts(self):
    print('ts:            ', self.ts)
    print('ts rgb:        ', self.rgb.ts)
    print('ts depth:      ', self.depth.ts)
    print('ts detection:  ', self.detections.ts)
    print('ts odom:       ', self.odom.ts)

class FrameRGBD:
  def __init__(self, *args): #rgb, depth
    if len(args) == 2:
      self.rgb = args[0]
      self.depth = args[1]
      self.ts = self.rgb.ts
    elif len(args) == 0:
      self.rgb = None
      self.depth = None
      self.ts = None
    else:
      raise ValueError('invalid number of arguments in FrameRGBD')
  def print_ts(self):
    print('ts:            ', self.ts)
    print('ts rgb:        ', self.rgb.ts)
    print('ts depth:      ', self.depth.ts)

class FrameRGBDWithOdom:
  def __init__(self, *args): #rgb, depth, pose
    if len(args) == 3:
      self.rgb = args[0]
      self.depth = args[1]
      self.odom = args[2]
      self.ts = self.rgb.ts
    elif len(args) == 0:
      self.rgb = None
      self.depth = None
      self.odom = None
      self.ts = None
    else:
      raise ValueError('invalid number of arguments in FrameRGBDWithPose')
  def print_ts(self):
    print('ts:            ', self.ts)
    print('ts rgb:        ', self.rgb.ts)
    print('ts depth:      ', self.depth.ts)
    print('ts pose:       ', self.odom.ts)

class Timestamp:
  def __init__(self, sec, nsec):
    self.sec = int(sec)
    self.nsec = int(nsec)

  def __eq__(self, other):
    return self.sec == other.sec and self.nsec == other.nsec
  
  def __lt__(self, other):
    if isinstance(other, Timestamp):
      return self.sec < other.sec or (self.sec == other.sec and self.nsec < other.nsec)
    elif isinstance(other, int):
      raise ValueError('Timestamp cannot be compared with int. Unit is ambiguous.')
    return self.sec < other.sec or (self.sec == other.sec and self.nsec < other.nsec)
  
  def __le__(self, other):
    return self < other or self == other
  
  def __gt__(self, other):
    return not self <= other
  
  def __repr__(self) -> str:
    return f'{self.sec}.{self.nsec:09d}'

  def __add__(self, other):
    sec = self.sec + other.sec
    nsec = self.nsec + other.nsec
    if nsec >= 1e9:
      sec += 1
      nsec -= 1e9
    return Timestamp(sec, nsec)

  def __sub__(self, other):
    sec = self.sec - other.sec
    nsec = self.nsec - other.nsec
    if nsec < 0:
      sec -= 1
      nsec += 1e9
    return Timestamp(sec, nsec)

  def toSeconds(self):
    return self.sec + self.nsec / 1e9

  def toMilliseconds(self):
    return self.sec * 1e3 + self.nsec / 1e6
  