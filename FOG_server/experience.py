from collections import namedtuple

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

