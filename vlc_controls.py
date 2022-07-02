import vlc

media_pth = "E:/media/August.Rush.FRENCH.DVDRiP.XviD-iD.AVI"
media = vlc.MediaPlayer(media_pth)


def control_vlc(command, media):

    if command == 'Play':
        media.play()

    # elif command == 'Pause':
    #     media.pause()

    elif command == 'Stop':
        media.stop()

    elif command == 'Volume_Up':
        current_vol = media.audio_get_volume()
        media.audio_set_volume(current_vol + 1)

    elif command == 'Volume_Down':
        current_vol = media.audio_get_volume()
        media.audio_set_volume(current_vol - 1)
