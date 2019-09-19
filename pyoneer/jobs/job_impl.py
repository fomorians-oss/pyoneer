from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import tensorflow as tf


class Job(object):
    """
    Manages a TensorFlow job consisting of checkpoints and summaries.

    Args:
        directory: A directory to save summaries and checkpoints to.
        max_to_keep: The maximum number of checkpoints to keep.
        keep_checkpoint_every_n_hours: Keep checkpoint every N hours.
        **kwargs: Checkpointable objects to save with the checkpoint.
    """

    def __init__(
        self, directory, max_to_keep=None, keep_checkpoint_every_n_hours=None, **kwargs
    ):
        self.directory = directory
        self.checkpoint = tf.train.Checkpoint(**kwargs)
        self.manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=directory,
            max_to_keep=max_to_keep,
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
        )
        self._summary_writers = {}

    def flush_summaries(self):
        for name in self._summary_writers.keys():
            summary_writer = self._summary_writers[name]
            summary_writer.flush()

    def summary_context(self, name=None, max_queue=None, flush_millis=None):
        """
        Gets or creates a summary writer context.

        Args:
            name: name for the summary writer.

        Returns:
            Summary writer context manager.
        """
        if name in self._summary_writers:
            summary_writer = self._summary_writers[name]
        else:
            summary_writer_path = os.path.join(self.directory, name)
            summary_writer = tf.summary.create_file_writer(
                summary_writer_path, max_queue=max_queue, flush_millis=flush_millis
            )
            self._summary_writers[name] = summary_writer
        return summary_writer.as_default()

    def save(self, checkpoint_number=None):
        """
        Create a new checkpoint.

        Args:
            checkpoint_number: An optional integer, or an integer-dtype Variable or Tensor,
                used to number the checkpoint. If None (default), checkpoints are numbered using
                checkpoint.save_counter. Even if checkpoint_number is provided, save_counter is
                still incremented. A user-provided checkpoint_number is not incremented even if
                it is a Variable.

        Returns:
            The path to the new checkpoint. It is also recorded in the `job.manager.checkpoints`
            and `job.manager.latest_checkpoint` properties.
        """
        return self.manager.save(checkpoint_number)

    def restore(self, path=None):
        """
        Restore a checkpoint.

        Args:
            path: Path to the checkpoint. (Default: `job.manager.latest_checkpoint`)

        Returns:
            The checkpoint restore status.
        """
        if path is None:
            path = self.manager.latest_checkpoint
        status = self.checkpoint.restore(path)
        return status
