import tensorflow.keras.backend as K
from tensorflow.python.keras.layers.recurrent import *


class RNN(RNN):
    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
            # The input_shape here could be a nest structure.

        # do the tensor_shape to shapes here. The input could be single tensor, or a
        # nested structure of tensors.
        x_input_shape = input_shape

        def get_input_spec(shape):
            if isinstance(shape, tensor_shape.TensorShape):
                input_spec_shape = shape.as_list()
            else:
                input_spec_shape = list(shape)
            batch_index, time_step_index = (1, 0) if self.time_major else (0, 1)
            if not self.stateful:
                input_spec_shape[batch_index] = None
            input_spec_shape[time_step_index] = None
            return InputSpec(shape=tuple(input_spec_shape))

        def get_step_input_shape(shape):
            if isinstance(shape, tensor_shape.TensorShape):
                shape = tuple(shape.as_list())
            # remove the timestep from the input_shape
            return shape[1:] if self.time_major else (shape[0],) + shape[2:]

        # Check whether the input shape contains any nested shapes. It could be
        # (tensor_shape(1, 2), tensor_shape(3, 4)) or (1, 2, 3) which is from numpy
        # inputs.
        try:
            input_shape = tensor_shape.as_shape(input_shape)
        except (ValueError, TypeError):
            # A nested tensor input
            pass

        if not nest.is_sequence(input_shape):
            # This indicates the there is only one input.
            if self.input_spec is not None:
                self.input_spec[0] = get_input_spec(input_shape)
            else:
                self.input_spec = [get_input_spec(input_shape)]
            step_input_shape = get_step_input_shape(input_shape)
        else:
            if self.input_spec is not None:
                self.input_spec[0] = nest.map_structure(get_input_spec, input_shape)
            else:
                self.input_spec = generic_utils.to_list(
                    nest.map_structure(get_input_spec, input_shape))
            step_input_shape = nest.map_structure(get_step_input_shape, input_shape)

        # allow cell (if layer) to build before we set or validate state_spec
        if isinstance(self.cell, Layer):
            if not self.cell.built:
                self.cell.build([step_input_shape, x_input_shape])

        # set or validate state_spec
        if _is_multiple_state(self.cell.state_size):
            state_size = list(self.cell.state_size)
        else:
            state_size = [self.cell.state_size]

        if self.state_spec is not None:
            # initial_state was passed in call, check compatibility
            self._validate_state_spec(state_size, self.state_spec)
        else:
            self.state_spec = [
                InputSpec(shape=[None] + tensor_shape.as_shape(dim).as_list())
                for dim in state_size
            ]
        if self.stateful:
            self.reset_states()
        self.built = True

    def call(self,
             inputs,
             mask=None,
             training=None,
             initial_state=None,
             constants=None):
        x_inputs = inputs
        # The input should be dense, padded with zeros. If a ragged input is fed
        # into the layer, it is padded and the row lengths are used for masking.
        inputs, row_lengths = K.convert_inputs_if_ragged(inputs)
        is_ragged_input = (row_lengths is not None)
        self._validate_args_if_ragged(is_ragged_input, mask)

        inputs, initial_state, constants = self._process_inputs(
            inputs, initial_state, constants)

        self._maybe_reset_cell_dropout_mask(self.cell)
        if isinstance(self.cell, StackedRNNCells):
            for cell in self.cell.cells:
                self._maybe_reset_cell_dropout_mask(cell)

        if mask is not None:
            # Time step masks must be the same for each input.
            # TODO(scottzhu): Should we accept multiple different masks?
            mask = nest.flatten(mask)[0]

        if nest.is_sequence(inputs):
            # In the case of nested input, use the first element for shape check.
            input_shape = K.int_shape(nest.flatten(inputs)[0])
        else:
            input_shape = K.int_shape(inputs)
        timesteps = input_shape[0] if self.time_major else input_shape[1]
        if self.unroll and timesteps is None:
            raise ValueError('Cannot unroll a RNN if the '
                             'time dimension is undefined. \n'
                             '- If using a Sequential model, '
                             'specify the time dimension by passing '
                             'an `input_shape` or `batch_input_shape` '
                             'argument to your first layer. If your '
                             'first layer is an Embedding, you can '
                             'also use the `input_length` argument.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a `shape` '
                             'or `batch_shape` argument to your Input layer.')

        kwargs = {}
        if generic_utils.has_arg(self.cell.call, 'training'):
            kwargs['training'] = training

        # TF RNN cells expect single tensor as state instead of list wrapped tensor.
        is_tf_rnn_cell = getattr(self.cell, '_is_tf_rnn_cell', None) is not None
        if constants:
            if not generic_utils.has_arg(self.cell.call, 'constants'):
                raise ValueError('RNN cell does not support constants')

            def step(inputs, states):
                constants = states[-self._num_constants:]  # pylint: disable=invalid-unary-operand-type
                states = states[:-self._num_constants]  # pylint: disable=invalid-unary-operand-type

                states = states[0] if len(states) == 1 and is_tf_rnn_cell else states
                output, new_states = self.cell.call(
                    inputs, states, x_inputs, constants=constants, **kwargs)
                if not nest.is_sequence(new_states):
                    new_states = [new_states]
                return output, new_states
        else:

            def step(inputs, states):
                states = states[0] if len(states) == 1 and is_tf_rnn_cell else states
                output, new_states = self.cell.call(inputs, states, x_inputs, **kwargs)
                if not nest.is_sequence(new_states):
                    new_states = [new_states]
                return output, new_states

        last_output, outputs, states = K.rnn(
            step,
            inputs,
            initial_state,
            constants=constants,
            go_backwards=self.go_backwards,
            mask=mask,
            unroll=self.unroll,
            input_length=row_lengths if row_lengths is not None else timesteps,
            time_major=self.time_major,
            zero_output_for_mask=self.zero_output_for_mask)

        if self.stateful:
            updates = []
            for state_, state in zip(nest.flatten(self.states), nest.flatten(states)):
                updates.append(state_ops.assign(state_, state))
            self.add_update(updates)

        if self.return_sequences:
            output = K.maybe_convert_to_ragged(is_ragged_input, outputs, row_lengths)
        else:
            output = last_output

        if self.return_state:
            if not isinstance(states, (list, tuple)):
                states = [states]
            else:
                states = list(states)
            return generic_utils.to_list(output) + states
        else:
            return output


def _time_distributed_dense(x, w, b=None, dropout=None,
                            input_dim=None, output_dim=None,
                            timesteps=None, training=None):
    """Apply `y . w + b` for every temporal slice y of x.
    # Arguments
        x: input tensor.
        w: weight matrix.
        b: optional bias vector.
        dropout: wether to apply dropout (same dropout mask
            for every temporal slice of the input).
        input_dim: integer; optional dimensionality of the input.
        output_dim: integer; optional dimensionality of the output.
        timesteps: integer; optional number of timesteps.
        training: training phase tensor or boolean.
    # Returns
        Output tensor.
    """
    if not input_dim:
        input_dim = K.shape(x)[2]
    if not timesteps:
        timesteps = K.shape(x)[1]
    if not output_dim:
        output_dim = K.int_shape(w)[1]

    if dropout is not None and 0. < dropout < 1.:
        # apply the same dropout pattern at every timestep
        ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
        dropout_matrix = K.dropout(ones, dropout)
        expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
        x = K.in_train_phase(x * expanded_dropout_matrix, x, training=training)

    # collapse time dimension and batch dimension together
    x = K.reshape(x, (-1, input_dim))
    x = K.dot(x, w)
    if b is not None:
        x = K.bias_add(x, b)
    # reshape to 3D tensor
    if K.backend() == 'tensorflow':
        x = K.reshape(x, K.stack([-1, timesteps, output_dim]))
        x.set_shape([None, None, output_dim])
    else:
        x = K.reshape(x, (-1, timesteps, output_dim))
    return x


class AttentionLSTMCell(DropoutRNNCellMixin, Layer):
    """Cell class for the LSTM layer.

    Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      recurrent_activation: Activation function to use
        for the recurrent step.
        Default: hard sigmoid (`hard_sigmoid`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix,
        used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      unit_forget_bias: Boolean.
        If True, add 1 to the bias of the forget gate at initialization.
        Setting it to true will also force `bias_initializer="zeros"`.
        This is recommended in [Jozefowicz et
          al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
        the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
        the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the recurrent state.
      implementation: Implementation mode, either 1 or 2.
        Mode 1 will structure its operations as a larger number of
        smaller dot products and additions, whereas mode 2 will
        batch them into fewer, larger operations. These modes will
        have different performance profiles on different hardware and
        for different applications.

    Call arguments:
      inputs: A 2D tensor.
      states: List of state tensors corresponding to the previous timestep.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. Only relevant when `dropout` or
        `recurrent_dropout` is used.
    """

    def __init__(self,
                 units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 attention_activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 attention_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 attention_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 attention_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 **kwargs):
        self._enable_caching_device = kwargs.pop('enable_caching_device', False)
        super(AttentionLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.attention_activation = activations.get(attention_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.attention_initializer = initializers.get(attention_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.attention_regularizer = regularizers.get(attention_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.attention_constraint = constraints.get(attention_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        if self.recurrent_dropout != 0 and implementation != 1:
            logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
            self.implementation = 1
        else:
            self.implementation = implementation
        # tuple(_ListWrapper) was silently dropping list content in at least 2.7.10,
        # and fixed after 2.7.16. Converting the state_size to wrapper around
        # NoDependency(), so that the base_layer.__setattr__ will not convert it to
        # ListWrapper. Down the stream, self.states will be a list since it is
        # generated from nest.map_structure with list, and tuple(list) will work
        # properly.
        self.state_size = data_structures.NoDependency([self.units, self.units])
        self.output_size = self.units

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_shape, x_input_shape = input_shape
        default_caching_device = _caching_device(self)
        self.input_dim = input_dim = input_shape[-1]
        self.timestep_dim = x_input_shape[1]

        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            caching_device=default_caching_device)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            caching_device=default_caching_device)

        self.attention_kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name='attention_kernel',
            initializer=self.attention_initializer,
            regularizer=self.attention_regularizer,
            constraint=self.attention_constraint)

        # add attention weights
        # weights for attention model
        self.attention_weights = self.add_weight(shape=(input_dim, self.units),
                                                 name='attention_W',
                                                 initializer=self.attention_initializer,
                                                 regularizer=self.attention_regularizer,
                                                 constraint=self.attention_constraint)

        self.attention_recurrent_weights = self.add_weight(shape=(self.units, self.units),
                                                           name='attention_U',
                                                           initializer=self.recurrent_initializer,
                                                           regularizer=self.recurrent_regularizer,
                                                           constraint=self.recurrent_constraint)

        if self.use_bias:
            if self.unit_forget_bias:

                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(
                shape=(self.units * 4,),
                name='bias',
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                caching_device=default_caching_device)
            self.attention_bias = self.add_weight(shape=(self.units,),
                                                  name='attention_b',
                                                  initializer=self.bias_initializer,
                                                  regularizer=self.bias_regularizer,
                                                  constraint=self.bias_constraint)
            self.attention_recurrent_bias = self.add_weight(shape=(self.units, 1),
                                                            name='attention_v',
                                                            initializer=self.bias_initializer,
                                                            regularizer=self.bias_regularizer,
                                                            constraint=self.bias_constraint)
        else:
            self.bias = None
            self.attention_bias = None
            self.attention_recurrent_bias = None

        self.built = True

    def _compute_carry_and_output(self, x, h_tm1, c_tm1, z_hat):
        """Computes carry and output using split kernels."""
        x_i, x_f, x_c, x_o = x
        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
        i = self.recurrent_activation(
            x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]) + K.dot(z_hat, self.attention_i))
        f = self.recurrent_activation(x_f + K.dot(
            h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]) + K.dot(z_hat, self.attention_f))
        c = f * c_tm1 + i * self.activation(x_c + K.dot(
            h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]) + K.dot(z_hat, self.attention_c))
        o = self.recurrent_activation(
            x_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]) + K.dot(z_hat, self.attention_o))
        return c, o

    def _compute_carry_and_output_fused(self, z, c_tm1):
        """Computes carry and output using fused kernels."""
        z0, z1, z2, z3 = z
        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.recurrent_activation(z3)
        return c, o

    def call(self, inputs, states, x_input, training=None):
        self.attention_i = self.attention_kernel[:, :self.units]
        self.attention_f = self.attention_kernel[:, self.units: self.units * 2]
        self.attention_c = self.attention_kernel[:, self.units * 2: self.units * 3]
        self.attention_o = self.attention_kernel[:, self.units * 3:]

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=4)

        h_att = K.repeat(h_tm1, self.timestep_dim)
        att = _time_distributed_dense(x_input, self.attention_weights,
                                      self.attention_bias, output_dim=K.int_shape(self.attention_weights)[1])
        attention_ = self.attention_activation(K.dot(h_att, self.attention_recurrent_weights) + att)
        attention_ = K.squeeze(K.dot(attention_, self.attention_recurrent_bias), 2)

        alpha = K.exp(attention_)

        # if 0 < self.dropout < 1.:
        #     alpha *= dp_mask[0]

        alpha /= K.sum(alpha, axis=1, keepdims=True)
        alpha_r = K.repeat(alpha, self.input_dim)
        alpha_r = K.permute_dimensions(alpha_r, (0, 2, 1))

        # make context vector (soft attention after Bahdanau et al.)
        z_hat = x_input * alpha_r
        # context_sequence = z_hat
        z_hat = K.sum(z_hat, axis=1)

        if self.implementation == 1:
            if 0 < self.dropout < 1.:
                inputs_i = inputs * dp_mask[0]
                inputs_f = inputs * dp_mask[1]
                inputs_c = inputs * dp_mask[2]
                inputs_o = inputs * dp_mask[3]
            else:
                inputs_i = inputs
                inputs_f = inputs
                inputs_c = inputs
                inputs_o = inputs

            k_i, k_f, k_c, k_o = array_ops.split(
                self.kernel, num_or_size_splits=4, axis=1)
            x_i = K.dot(inputs_i, k_i)
            x_f = K.dot(inputs_f, k_f)
            x_c = K.dot(inputs_c, k_c)
            x_o = K.dot(inputs_o, k_o)
            if self.use_bias:
                b_i, b_f, b_c, b_o = array_ops.split(
                    self.bias, num_or_size_splits=4, axis=0)
                x_i = K.bias_add(x_i, b_i)
                x_f = K.bias_add(x_f, b_f)
                x_c = K.bias_add(x_c, b_c)
                x_o = K.bias_add(x_o, b_o)

            if 0 < self.recurrent_dropout < 1.:
                h_tm1_i = h_tm1 * rec_dp_mask[0]
                h_tm1_f = h_tm1 * rec_dp_mask[1]
                h_tm1_c = h_tm1 * rec_dp_mask[2]
                h_tm1_o = h_tm1 * rec_dp_mask[3]
            else:
                h_tm1_i = h_tm1
                h_tm1_f = h_tm1
                h_tm1_c = h_tm1
                h_tm1_o = h_tm1
            x = (x_i, x_f, x_c, x_o)
            h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
            c, o = self._compute_carry_and_output(x, h_tm1, c_tm1, z_hat)
        else:
            if 0. < self.dropout < 1.:
                inputs = inputs * dp_mask[0]
            if 0. < self.recurrent_dropout < 1.:
                h_tm1 = h_tm1 * rec_dp_mask[0]
            z = K.dot(inputs, self.kernel)
            z += K.dot(h_tm1, self.recurrent_kernel)
            z += K.dot(z_hat, self.attention_kernel)
            if self.use_bias:
                z = K.bias_add(z, self.bias)

            z = array_ops.split(z, num_or_size_splits=4, axis=1)
            c, o = self._compute_carry_and_output_fused(z, c_tm1)

        h = o * self.activation(c)
        return h, [h, c]

    def get_config(self):
        config = {
            'units':
                self.units,
            'activation':
                activations.serialize(self.activation),
            'recurrent_activation':
                activations.serialize(self.recurrent_activation),
            'attention_activation':
                activations.serialize(self.attention_activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'attention_initializer': initializers.serialize(self.attention_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'unit_forget_bias':
                self.unit_forget_bias,
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'attention_regularizer': regularizers.serialize(self.attention_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'attention_constraint': constraints.serialize(self.attention_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout,
            'implementation':
                self.implementation
        }
        base_config = super(AttentionLSTMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return list(_generate_zero_filled_state_for_cell(
            self, inputs, batch_size, dtype))


def _generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype):
    if inputs is not None:
        batch_size = array_ops.shape(inputs)[0]
        dtype = inputs.dtype
    return _generate_zero_filled_state(batch_size, cell.state_size, dtype)


def _generate_zero_filled_state(batch_size_tensor, state_size, dtype):
    """Generate a zero filled tensor with shape [batch_size, state_size]."""
    if batch_size_tensor is None or dtype is None:
        raise ValueError(
            'batch_size and dtype cannot be None while constructing initial state: '
            'batch_size={}, dtype={}'.format(batch_size_tensor, dtype))

    def create_zeros(unnested_state_size):
        flat_dims = tensor_shape.as_shape(unnested_state_size).as_list()
        init_state_size = [batch_size_tensor] + flat_dims
        return array_ops.zeros(init_state_size, dtype=dtype)

    if nest.is_sequence(state_size):
        return nest.map_structure(create_zeros, state_size)
    else:
        return create_zeros(state_size)


def _caching_device(rnn_cell):
    """Returns the caching device for the RNN variable.

    This is useful for distributed training, when variable is not located as same
    device as the training worker. By enabling the device cache, this allows
    worker to read the variable once and cache locally, rather than read it every
    time step from remote when it is needed.

    Note that this is assuming the variable that cell needs for each time step is
    having the same value in the forward path, and only gets updated in the
    backprop. It is true for all the default cells (SimpleRNN, GRU, LSTM). If the
    cell body relies on any variable that gets updated every time step, then
    caching device will cause it to read the stall value.

    Args:
      rnn_cell: the rnn cell instance.
    """
    if context.executing_eagerly():
        # caching_device is not supported in eager mode.
        return None
    if not getattr(rnn_cell, '_enable_caching_device', False):
        return None
    # Don't set a caching device when running in a loop, since it is possible that
    # train steps could be wrapped in a tf.while_loop. In that scenario caching
    # prevents forward computations in loop iterations from re-reading the
    # updated weights.
    if control_flow_util.IsInWhileLoop(ops.get_default_graph()):
        logging.warn('Variable read device caching has been disabled because the '
                     'RNN is in tf.while_loop loop context, which will cause '
                     'reading stalled value in forward path. This could slow down '
                     'the training due to duplicated variable reads. Please '
                     'consider updating your code to remove tf.while_loop if '
                     'possible.')
        return None
    if rnn_cell._dtype_policy.should_cast_variables:
        logging.warn('Variable read device caching has been disabled since it '
                     'doesn\'t work with the mixed precision API. This is '
                     'likely to cause a slowdown for RNN training due to '
                     'duplicated read of variable for each timestep, which '
                     'will be significant in a multi remote worker setting. '
                     'Please consider disabling mixed precision API if '
                     'the performance has been affected.')
        return None
    # Cache the value on the device that access the variable.
    return lambda op: op.device

def _is_multiple_state(state_size):
  """Check whether the state_size contains multiple states."""
  return (hasattr(state_size, '__len__') and
          not isinstance(state_size, tensor_shape.TensorShape))
