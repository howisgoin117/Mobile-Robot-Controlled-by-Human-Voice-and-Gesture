import pyaudio

def main():
    print("Initializing PyAudio...")
    p = pyaudio.PyAudio()
    
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    
    print("\n--- Available Audio Input Devices ---")
    for i in range(0, numdevices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if device_info.get('maxInputChannels') > 0:
            name = device_info.get('name')
            print(f"[{i}] {name}")
            
    print("-------------------------------------")
    print("Find your bluetooth headset above and note the index number [X].")
    print("You can then run the voice node with: --ros-args -p mic_index:=X")
    p.terminate()

if __name__ == '__main__':
    main()
