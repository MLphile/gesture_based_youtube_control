import vlc

media_pth = "E:/media/August.Rush.FRENCH.DVDRiP.XviD-iD.AVI"
media = vlc.MediaPlayer(media_pth)


def control_vlc(command, media):
    vol_range = range(0, 501, 1)

    if command == 'Play':
        return media.play()

    # elif command == 'Pause':
    #     return media.pause()

    elif command == 'Stop':
        return media.stop()
