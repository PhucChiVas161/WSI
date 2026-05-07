import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add the directory containing toupcam.dll to the system PATH
# Win
if os.name == 'nt':
    dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "win", "x64"))
    os.add_dll_directory(dll_path)


import toupcam
import cv2
import numpy as np

class App:
    def __init__(self):
        self.hcam = None
        self.buf = None
        self.total = 0
        self.width = None
        self.height = None

# the vast majority of callbacks come from toupcam.dll/so/dylib internal threads
    @staticmethod
    def cameraCallback(nEvent, ctx):
        if nEvent == toupcam.TOUPCAM_EVENT_IMAGE:
            ctx.CameraCallback(nEvent)

    def CameraCallback(self, nEvent):
        if nEvent == toupcam.TOUPCAM_EVENT_IMAGE:
            try:
                self.hcam.PullImageV4(self.buf, 0, 24, 0, None)
                self.total += 1
                
                # Convert buffer to numpy array and display
                img = np.frombuffer(self.buf, dtype=np.uint8).reshape((self.height, self.width, 3))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize to 1280x720 for display
                display_img = cv2.resize(img, (1280, 720))
                
                # Display the frame
                cv2.imshow('Camera Preview - 1280x720', display_img)
                cv2.waitKey(1)
            except toupcam.HRESULTException as ex:
                print('pull image failed, hr=0x{:x}'.format(ex.hr & 0xffffffff))
        else:
            print('event callback: {}'.format(nEvent))

    def run(self):
        a = toupcam.Toupcam.EnumV2()
        if len(a) > 0:
            print('{}: flag = {:#x}, preview = {}, still = {}'.format(a[0].displayname, a[0].model.flag, a[0].model.preview, a[0].model.still))
            for r in a[0].model.res:
                print('\t = [{} x {}]'.format(r.width, r.height))
            self.hcam = toupcam.Toupcam.Open(a[0].id)
            if self.hcam:
                try:
                    # Get the current camera resolution
                    width, height = self.hcam.get_Size()
                    self.width = width
                    self.height = height
                    bufsize = toupcam.TDIBWIDTHBYTES(width * 24) * height
                    print('image size: {} x {}, bufsize = {}'.format(width, height, bufsize))
                    self.buf = bytes(bufsize)
                    if self.buf:
                        try:
                            self.hcam.StartPullModeWithCallback(self.cameraCallback, self)
                            print('Camera preview started. Press ENTER to exit or close the preview window.')
                        except toupcam.HRESULTException as ex:
                            print('failed to start camera, hr=0x{:x}'.format(ex.hr & 0xffffffff))
                    input('press ENTER to exit')
                    cv2.destroyAllWindows()
                finally:
                    self.hcam.Close()
                    self.hcam = None
                    self.buf = None
            else:
                print('failed to open camera')
        else:
            print('no camera found')

if __name__ == '__main__':
    app = App()
    app.run()