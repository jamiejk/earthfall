"""Minimal clay layout: adult proportions and unchanged original-clock camera."""
import math

from touchdown.shoreline_camera import START, END, FRAMES, FPS, camera_pose, ground_height, shore

HEIGHT = 1.78
HEAD_HEIGHT = .23
KEYFRAMES = ((0, 'regional'), (12, 'regional-exit'), (36, 'approach'),
             (60, 'settle'), (96, 'finish'))


def adult_landmarks(frame):
    if not START <= frame <= END:
        raise ValueError('Outside original camera segment')
    z = ground_height(0, 0)
    u = max(0., min(1., (frame-156)/36))
    u = u*u*(3-2*u)
    angle = .08+1.38*u
    points = dict(head=(0, 0, z+HEIGHT-HEAD_HEIGHT/2), neck=(0, 0, z+1.535),
                  torso=(0, 0, z+1.245), pelvis=(0, 0, z+.95))
    for side, sign in [('left', -1), ('right', 1)]:
        shoulder = (sign*.20, 0, z+1.46)
        elbow = (shoulder[0]+sign*.29*math.sin(angle), 0, shoulder[2]-.29*math.cos(angle))
        wrist = (elbow[0]+sign*.25*math.sin(angle+.12), -.025, elbow[2]-.25*math.cos(angle+.12))
        points.update({f'{side}_shoulder': shoulder, f'{side}_elbow': elbow,
            f'{side}_wrist': wrist, f'{side}_hand': (wrist[0]+sign*.035*math.sin(angle+.12), -.025, wrist[2]-.035*math.cos(angle+.12)),
            f'{side}_hip': (sign*.09, 0, z+.91), f'{side}_knee': (sign*.10, -.012, z+.51),
            f'{side}_ankle': (sign*.11, 0, z+.09), f'{side}_foot': (sign*.11, -.055, z+.04)})
    return points
