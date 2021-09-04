from gym.envs.registration import register 
register(
	id='slug-v0',
	entry_point='gym_slug.envs:UnbreakableSeaweed',) 

register(
	id='slug-v1',
	entry_point='gym_slug.envs:BreakableSeaweed',)