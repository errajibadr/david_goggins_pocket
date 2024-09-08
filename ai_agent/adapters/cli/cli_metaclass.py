import click
import copy
import inspect
import warnings


class ClickInstantiator:
    klass = None
    command = None

    def __init__(self, command, klass):
        self.command = command
        self.klass = klass

    def __call__(self, *args, **kwargs):
        return self.command(self.klass(), *args, **kwargs)


class ClickCommandMetaclass(type):
    def __new__(mcs, name, bases, dct):
        klass = super().__new__(mcs, name, bases, dct)

        # create and populate the click.Group for this Class
        klass.click_group = click.Group(name=klass.__name__.lower())

        # warn about @click.command decorators missing the parens
        for name, command in inspect.getmembers(klass, inspect.isfunction):
            if repr(command).startswith("<function command."):
                warnings.warn(
                    "%s.%s is wrapped with click.command without parens, please add them"
                    % (klass.__name__, name)
                )

        for name, command in inspect.getmembers(
            klass, lambda x: isinstance(x, click.Command)
        ):
            if name == "click_group":
                continue

            def find_final_command(target):
                """Find the last call command at the end of a stack of click.Command instances"""
                while isinstance(target.callback, click.Command):
                    target = target.callback
                return target

            command_target = find_final_command(command)

            if not isinstance(command_target.callback, ClickInstantiator):
                # the top class to implement this
                command_target.callback = ClickInstantiator(
                    command_target.callback, klass
                )
            else:
                # this is a subclass function, copy it and replace the klass
                setattr(klass, name, copy.deepcopy(command))
                command = getattr(klass, name)
                find_final_command(getattr(klass, name)).callback.klass = klass

            # now add it to the group
            klass.click_group.add_command(command, name)
        return klass


class ClickCommandBase(metaclass=ClickCommandMetaclass):
    pass
