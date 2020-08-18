from types import MappingProxyType


class Curriculum:
    def __init__(self, level=0):
        self.level = level

    def update(self):
        raise NotImplementedError()

    def get(self, attribute):
        raise NotImplementedError()


class Manual_Curriculum(Curriculum):
    def __init__(self, configuration_file, level=0):
        super(Manual_Curriculum, self).__init__(level=level)
        self.file = configuration_file
        import yaml
        with open(configuration_file) as f:
            self.curriculum = yaml.safe_load(f)
        print("Manual curriculum: ", self.curriculum)
        self.num_levels = len(self.curriculum["curriculum"])

    def update(self):
        # Increase level
        if self.level == self.num_levels - 1:
            raise StopIteration()
        self.level += 1

    def get(self, attribute):
        return self.curriculum["curriculum"][self.level][attribute]


def offset_curriculum_generator(num_levels,
                                initial_values,
                                offset=MappingProxyType({"x_dim": lambda lvl: 0.75,
                                                         "y_dim": lambda lvl: 0.5,
                                                         "n_agents": lambda lvl: 0.1,
                                                         "n_cities": lambda lvl: 0.13,
                                                         # "n_extra": lambda lvl: 0.2,
                                                         # "min_dist": lambda lvl: 0.3,
                                                         # "max_dist": lambda lvl: 0.3,
                                                         "max_rails_between_cities": lambda lvl: 0.05,
                                                         "max_rails_in_city": lambda lvl: 0.05,
                                                         }),
                                forget_every=None,
                                forget_intensity=None,
                                checkpoint_every=20,
                                checkpoint_recovery_levels=5):
    for k in initial_values:
        initial_values[k] = float(initial_values[k])

    checkpoint = initial_values
    checkpoint_counter = 0

    for level in range(1, num_levels + 1):

        if checkpoint_every is not None and checkpoint_recovery_levels is not None:
            # Save checkpoint
            if level % checkpoint_every and not checkpoint_counter > 0:
                checkpoint, initial_values = initial_values, checkpoint
                checkpoint_counter = checkpoint_recovery_levels

            # Solve checkpoints
            if checkpoint_counter > 0:
                initial_values = {k: (initial_values[k] + offset[k](level)) for k, v in initial_values.items()}
                checkpoint_counter -= 1

        for k, v in initial_values.items():
            if forget_every is not None and level % forget_every == 0 and forget_intensity is not None:
                initial_values[k] = v + offset[k](level) - forget_intensity
            else:
                initial_values[k] = v + offset[k](level)
        yield initial_values


class Semi_Auto_Curriculum(Curriculum):
    def __init__(self, generator, num_levels, level=0):
        super(Semi_Auto_Curriculum, self).__init__(level=level)
        self.generator = generator
        self.values = next(generator)
        self.num_levels = num_levels

    def update(self):
        """

        :return:

        :raise StopIteration
        """
        # Increase level in the iterator
        self.values = next(self.generator)

    def get(self, attribute):
        return int(self.values[attribute])
