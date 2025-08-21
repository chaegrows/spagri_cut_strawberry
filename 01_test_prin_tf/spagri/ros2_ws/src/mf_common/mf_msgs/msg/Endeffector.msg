std_msgs/Header header

uint16    sw_version
uint16    len_frame
uint8    n_position
float32[] positions # len: 16, valid: val:=int(float*100), invalid: undefined
uint8    n_digital_io
uint16[]  digital_ios # len: 16, valid: val:=val, invalid: undefined
uint32    event

uint32 EVENT_NONE=4294967295
uint16 DEV_HOST_PC=10000
uint16 DEV_POLLINATOR=200
uint16 DEV_HARVESTOR=100
