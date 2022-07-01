import vlc

media_pth = "E:/media/August.Rush.FRENCH.DVDRiP.XviD-iD.AVI"
media = vlc.MediaPlayer(media_pth)


def control_vlc(command, media):
    min_vol = 0
    max_vol = 401

    if command == 'Play':
        media.play()

    # elif command == 'Pause':
    #     media.pause()

    elif command == 'Stop':
        media.stop()

    elif command == 'Unmute':
        current_vol = media.audio_get_volume()
        for i in range(current_vol, max_vol, 1):
            media.audio_set_volume(i)

    elif command == 'Mute':
        current_vol = media.audio_get_volume()
        for i in range(current_vol, min_vol, - 1):
            media.audio_set_volume(i)
    elif command == 'Middle_Vol':
        media.audio_set_volume(200)
