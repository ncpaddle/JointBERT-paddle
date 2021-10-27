from typing import List, Optional
import paddle
import paddle.nn as nn


class CRF(nn.Layer):
    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first

        self.start_transitions = paddle.create_parameter(shape=[num_tags], dtype=paddle.float32,
                                                                default_initializer=paddle.nn.initializer.Uniform(low=- 0.1, high=0.1))
        self.end_transitions = paddle.create_parameter(shape=[num_tags], dtype=paddle.float32,
                                                                default_initializer=paddle.nn.initializer.Uniform(low=- 0.1, high=0.1))
        self.transitions = paddle.create_parameter(shape=[num_tags, num_tags], dtype=paddle.float32,
                                                          default_initializer=paddle.nn.initializer.Uniform(low=- 0.1, high=0.1))


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(
            self,
            emissions,
            tags,
            mask=None,
            reduction: str = 'sum',
    ):

        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = paddle.ones_like(tags, dtype=paddle.int32)

        if self.batch_first:
            emissions = emissions.transpose([1, 0, 2])
            tags = tags.transpose([1, 0])
            mask = mask.transpose([1, 0])

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.astype(paddle.float32).sum()


    def decode(self, emissions,
               mask=None) -> List[List[int]]:

        self._validate(emissions, mask=mask)
        if mask is None:
            mask = paddle.ones(shape=emissions.shape[:2], dtype=paddle.int32)

        if self.batch_first:
            emissions = emissions.transpose([1, 0, 2])
            mask = mask.transpose([1, 0])

        return self._viterbi_decode(emissions, mask)


    def _validate(
            self,
            emissions,
            tags=None,
            mask=None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.shape[2] != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.shape[2]}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].astype(paddle.bool).all().astype(paddle.uint8)
            no_empty_seq_bf = self.batch_first and mask[:, 0].astype(paddle.bool).all().astype(paddle.uint8)
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(
            self, emissions, tags, mask):
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.shape[2] == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].astype(paddle.bool).all().astype(paddle.uint8)

        seq_length, batch_size = tags.shape
        mask = mask.astype(paddle.float32)

        # Start transition score and first emission
        # shape: (batch_size,)

        score = self.start_transitions.index_select(index=tags[0])
        score += emissions[0].gather_nd(paddle.stack(
            [paddle.arange(batch_size), tags[0]], axis=-1
        ))

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions.gather_nd(paddle.stack(
                [tags[i-1], tags[i]],
                axis=-1
            )) * mask[i]
            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i].gather_nd(paddle.stack(
                [paddle.arange(batch_size), tags[i]],
                axis=-1
            )) * mask[i]
            # score += emissions[i, 0:batch_size].index_select(index=tags[i], axis=-1) * mask[i]

        # End transition score
        # shape: (batch_size,)

        seq_ends = mask.astype(paddle.int64).sum(axis=0) - 1
        # shape: (batch_size,)
        last_tags = tags.gather_nd(paddle.stack(
            [seq_ends, paddle.arange(batch_size)],
            axis=-1
        ))

        # shape: (batch_size,)
        # score += self.end_transitions[last_tags]
        score += self.end_transitions.index_select(index=last_tags)
        # score += paddle.to_tensor(self.end_transitions.numpy()[last_tags.numpy()])

        return score

    def _compute_normalizer(
            self, emissions, mask):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.shape[2] == self.num_tags
        assert mask[0].astype(paddle.bool).all()

        seq_length = emissions.shape[0]

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = paddle.logsumexp(next_score, axis=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            condition = mask[i].unsqueeze(1).astype(paddle.bool).expand(shape=next_score.shape).astype(paddle.bool)
            score = paddle.where(condition, next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return paddle.logsumexp(score, axis=1)

    def _viterbi_decode(self, emissions,
                        mask) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.shape[2] == self.num_tags
        assert mask.astype(paddle.int64)[0].astype(paddle.bool).all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            # next_score, indices = next_score.max(axis=1)
            next_score, indices = next_score.max(axis=1), next_score.argmax(axis=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            condition = mask[i].unsqueeze(1).expand(shape=next_score.shape).astype(paddle.bool)
            score = paddle.where(condition, next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.astype(paddle.int64).sum(axis=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            # _, best_last_tag = score[idx].max(axis=0)
            best_last_tag = score[idx].argmax(axis=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list