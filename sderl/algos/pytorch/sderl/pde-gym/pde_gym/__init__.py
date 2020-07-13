from gym.envs.registration import register

register(
    id='allencahn-v0',
    entry_point='pde_gym.envs:AllenCahn',
)
register(
    id='hjb-v0',
    entry_point='pde_gym.envs:HJB',
)
register(
    id='pricingoption-v0',
    entry_point='pde_gym.envs:PricingOption'
)
