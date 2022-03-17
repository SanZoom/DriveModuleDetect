import serial


class SerialPort:
    def __init__(self, _port='/dev/ttyUSB0', speed=115200, _bytesize=8, _parity='N', stopbits=1):
        self.__serialPort = serial.Serial(port=_port,
                                          baudrate=speed,
                                          bytesize=_bytesize,
                                          parity=_parity,
                                          stopbits=stopbits)

        self.send_data = [0xFE, 0x00, 0x00, 0xFD]

    """
        both_hands_leaving_wheel
        eye_closed
        no_face_mask
        not_buckling_up
        smoke
        not_facing_front
        cellphone
        yawn
        head_down
        head_side
    """

    def send(self, person_state, pose_result):
        self.__reset()
        self.send_data[1] |= 0x01 << 7

        if person_state['both_hands_leaving_wheel']:
            self.send_data[1] |= 0x01 << 6
        if pose_result['eye_closed']:
            self.send_data[1] |= 0x01 << 5
        if person_state['no_face_mask']:
            self.send_data[1] |= 0x01 << 4
        if person_state['not_buckling_up']:
            self.send_data[1] |= 0x01 << 3
        if person_state['smoke']:
            self.send_data[1] |= 0x01 << 2
        if pose_result['not_facing_front']:
            self.send_data[1] |= 0x01 << 1
        if person_state['cellphone']:
            self.send_data[1] |= 0x01
        if pose_result['yawn']:
            self.send_data[2] |= 0x01 << 7
        if pose_result['head_down']:
            self.send_data[2] |= 0x01 << 6
        if pose_result['head_side']:
            self.send_data[2] |= 0x01 << 5

        self.__serialPort.write(bytearray(self.send_data))
        print(self.send_data)

    def __reset(self):
        self.send_data = [0xFE, 0x00, 0x00, 0xFD]