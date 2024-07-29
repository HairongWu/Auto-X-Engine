#include "autox_nn_ansi.h"


// avoid compling error: xxx_address will never be null
static int8_t condition_one(void* isa_op, void* naive_op) {
	return ((isa_op != NULL) && (naive_op != NULL));
}

static int8_t condition_two(void* isa_op, void* naive_op) {
	return ((isa_op == NULL) && (naive_op != NULL));
}

__m256 mul_ps_inline(__m256 a, __m256 b) {
	return _mm256_mul_ps(a, b);
}
__m256 add_ps_inline(__m256 a, __m256 b) {
	return _mm256_add_ps(a, b);
}
__m256 loadu_ps_inline(const float* a) {
	return _mm256_loadu_ps(a);
}

void storeu_ps_inline(float* b, __m256 a) {
	_mm256_storeu_ps(b, a);
}

__m256 set1_ps_inline(float a) {
	return _mm256_set1_ps(a);
}
float NaiveMul(float l, float r) {
	return l * r;
}
float NaiveAdd(float l, float r) {
	return l + r;
}

__m256 (*isa_op)(__m256, __m256) = &mul_ps_inline;
float (*naive_op)(float, float) = &NaiveMul;

void do_isa_elementwise(const float* dinx,
	const float* diny,
	float* dout,
	int num,
	int8_t IS_X_SINGLE,
	int8_t IS_Y_SINGLE) {

	int element_num = sizeof(__m256) / sizeof(float);
	int cnt = num / element_num;
	int remain = num % element_num;

	float* dinx_ptr = (float*)dinx;
	float* diny_ptr = (float*)diny;
	float* dout_ptr = dout;

	// avoid compiling error
	int8_t condition1 = condition_one((void*)isa_op,
		(void*)naive_op);
	int8_t condition2 = condition_two((void*)isa_op,
		(void*)naive_op);

	if (condition1) {
		__m256 rbx, rby;
		if (IS_X_SINGLE) {
			rbx = set1_ps_inline(*dinx);
		}
		if (IS_Y_SINGLE) {
			rby = set1_ps_inline(*diny);
		}

		for (int i = 0; i < cnt; i++) {
			__m256 dinx0, diny0, doutz0;
			if (!IS_X_SINGLE) {
				dinx0 = loadu_ps_inline((const float*)(dinx_ptr));
				dinx_ptr += element_num;
			}
			if (!IS_Y_SINGLE) {
				diny0 = loadu_ps_inline((const float*)(diny_ptr));
				diny_ptr += element_num;
			}
			if (IS_X_SINGLE && !IS_Y_SINGLE) {
				doutz0 = isa_op(rbx, diny0);
			}
			else if (!IS_X_SINGLE && IS_Y_SINGLE) {
				doutz0 = isa_op(dinx0, rby);
			}
			else if (!IS_X_SINGLE && !IS_Y_SINGLE) {
				doutz0 = isa_op(dinx0, diny0);
			}

			storeu_ps_inline((float*)(dout_ptr), doutz0);
			dout_ptr += element_num;
		}
		if (remain > 0) {
			for (int p = 0; p < remain; p++) {
				float tmp = naive_op(*dinx_ptr, *diny_ptr);
				
				*dout_ptr = tmp;
				dout_ptr++;
				if (!IS_X_SINGLE) {
					dinx_ptr++;
				}
				if (!IS_Y_SINGLE) {
					diny_ptr++;
				}
			}
		}
	}
	else if (condition2) {
		for (int p = 0; p < num; p++) {
			float tmp = naive_op(*dinx_ptr, *diny_ptr);
			
			*dout_ptr = tmp;
			dout_ptr++;
			if (!IS_X_SINGLE) {
				dinx_ptr++;
			}
			if (!IS_Y_SINGLE) {
				diny_ptr++;
			}
		}
	}
	else {
		//LOG(FATAL) << "do_isa_elementwise has no op function to call.";
	}
}

  void Elementwise_Broadcast(const float* dinx,                   
                                  const float* diny,                      
                                  float* dout,                            
                                  int batch,                          
                                  int channels,                       
                                  int num,                            
                                  int8_t inv) {                         
      for (int i = 0; i < batch; ++i) {                               
        for (int j = 0; j < channels; ++j) {                          
          int offset = (i * channels + j) * num;                      
          float* dout_ptr = dout + offset;                             
          if (inv) {                                                  
            const float* dinx_ptr = dinx + j;                          
            const float* diny_ptr = diny + offset;                     
			do_isa_elementwise(
                dinx_ptr, diny_ptr, dout_ptr, num, true, false);                   
          } else {                                                    
            const float* dinx_ptr = dinx + offset;                     
            const float* diny_ptr = diny + j;                          
			do_isa_elementwise(
                dinx_ptr, diny_ptr, dout_ptr, num, false, true);                   
          }                                                           
        }                                                             
      }                                                               
  }

// Remove trailing dimensions of size 1 for y
static uint16_t* trim_trailing_singular_dims(const uint16_t* dims, uint16_t *dims_size) {
	uint16_t actual_dims_size = *dims_size;
	for (; actual_dims_size != 0; --actual_dims_size) {
		if (dims[actual_dims_size - 1] != 1) break;
	}

	uint16_t* trim_dims = (uint16_t*)calloc(actual_dims_size, sizeof(uint16_t));
	for (int i = 0; i < actual_dims_size; ++i) {
		trim_dims[i] = dims[i];
	}
	if (actual_dims_size == 0) {
		*dims_size = 0;
		return NULL;
	}
	*dims_size = actual_dims_size;
	return trim_dims;
}

/*
 * Out = X point dot Y
 * If Y's shape does not match X' shape, they will be reshaped.
 * For example:
 * 1. shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
 *    pre=2, n=3*4, post=5
 *    x.shape(2, 12, 5) * y.shape(1, 12, 1).broadcast(2, 12, 5)
 * 2. shape(X) = (2, 3, 4, 5), shape(Y) = (4,5)
 *    pre=2*3, n=4*5, post=1
 *    x.shape(6, 20, 1) * y.shape(1, 20, 1).broadcast(6, 20, 1)
 * 3. force x_dims.size() is greater than y_dims.size(), else
 *    return false.
 */
int8_t is_fast_broadcast(const uint16_t* x_dims,
	const uint16_t* y_dims,
	int axis,
	uint16_t x_dims_size,
	uint16_t y_dims_size,
	int* pre,
	int* n,
	int* post) {
	if (axis == -1) {
		axis = x_dims_size - y_dims_size;
	}
	if (axis < 0) {
		// VLOG(4) << "Fast broadcast chk fail, for x_dims smaller.";
		return false;
	}
	uint16_t y_dim_trim_size = y_dims_size;
	uint16_t* y_dim_trim = trim_trailing_singular_dims(y_dims, &y_dim_trim_size);
	axis = (y_dim_trim_size == 0) ? x_dims_size : axis;
	if (x_dims_size < (y_dim_trim_size + axis)) {
		// VLOG(4) << "Fast broadcast chk fail, for y's shape size doesnt follow the "
		//	"axis rule";
		return false;
	}
	*pre = 1;
	*n = 1;
	*post = 1;
	for (int i = 0; i < axis; ++i) {
		(*pre) *= x_dims[i];
	}
	for (int i = 0; i < y_dim_trim_size; ++i) {
		if (x_dims[i + axis] != y_dim_trim[i]) {
			// VLOG(4) << "Fast broadcast chk fail, for dimension mismatch.";
			return false;
		}
		(*n) *= y_dim_trim[i];
	}
	for (int i = axis + y_dim_trim_size; i < x_dims_size; ++i) {
		(*post) *= x_dims[i];
	}
	return true;
}
int8_t same_dims(uint16_t* x_dims, uint16_t* y_dims, uint16_t x_dims_size,
	uint16_t y_dims_size)
{
	if (x_dims_size == y_dims_size)
	{
		for (int i = 0; i < x_dims_size; i++)
			if (x_dims[i] != y_dims[i])
				return false;
		return true;
	}
	return false;
}
/**
 * in Z = X op Y , there must be a minimal continuous mem in X or Y that could
 * do SIMD.
 */
enum BroadcastType {
	UNKNOWN,
	DIM_NOT_MATCH,  // could not do elementwise
	SAME_DIM,  // if x and y had a same dim, it could be treated as broadcast,but
	// not recommended.
	X_AS_CONTINUOUS,  // e.g. X.shape=[1,1,3,5],Y.shape=[2,4,1,1]
	Y_AS_CONTINUOUS,  // e.g. X.shape=[2,4,1,1],Y.shape=[1,1,3,5]
	BOTH_CONTINUOUS   // e.g. X.Shape=[1,1,3,5],Y.shape=[8,9,3,5]
};
/**
 * Get broadcast type, x_dims and x_dims must have same dim_size. The dimension
 * which will be broadcast should be set to 1, and the 1 at high dimension
 * should not be omitted
 * e.g. x_dims=[3,1,1,5] and y_dims=[1,2,4,1] is fine, but y_dims should not be
 * [2,4,1]
 * @tparam DimValue_t data type of dim's value
 * @param x_dims pointer to x's dim array, which is `dim_size` length
 * @param y_dims pointer to y's dim array, which is `dim_size` length
 * @param dim_size dim_size of x and y
 */
enum BroadcastType get_broadcast_type(const uint16_t *x_dims,
	const uint16_t *y_dims,
	const uint16_t *z_dims,
	int dim_size) {
	if (memcmp(x_dims, y_dims, sizeof(int32_t) * dim_size) == 0) {
		return SAME_DIM;
	}

	// check if it is broadcast
	for (int i = 0; i < dim_size; ++i) {
		if (x_dims[i] != 1 && y_dims[i] != 1 && x_dims[i] != y_dims[i]) {
			return DIM_NOT_MATCH;
		}
	}

	int pos = dim_size - 1;
	while (pos >= 0 && x_dims[pos] == y_dims[pos] && x_dims[pos] == 1) {
		if (z_dims[pos] != 1) {
			//LOG(FATAL) << "Unsupported broadcast type detected.";
			// Note: This is the 4th type of broadcast, It is not implemented to
			// reduce code complexity
			// e.g.
			// X.shape=[10,1],Y.shape=[10,1],Z.shape=[10,5] will match this pattern
			return DIM_NOT_MATCH;
		}
		--pos;
	}
	if (x_dims[pos] == y_dims[pos]) {
		return BOTH_CONTINUOUS;
	}
	if (x_dims[pos] != 1) {
		return X_AS_CONTINUOUS;
	}
	if (y_dims[pos] != 1) {
		return Y_AS_CONTINUOUS;
	}
	return UNKNOWN;
}
int8_t HasGapToNextDim(const uint16_t *dims,
	const uint16_t *stride,
	int this_dim){
	return (dims[this_dim + 1] * stride[this_dim + 1]) != stride[this_dim];
}
/**
 * fix missing dim of paddle lite tensor to fit this broadcast system.
 * @tparam DimValue_t
 * @param X
 * @param Y
 * @param Out
 * @param axis axis defined by paddle
 * @param out_dim_size dim size of Out
 * @param [out] p_x_dims fixed dim value of x
 * @param [out] p_y_dims fixed dim value of y
 */

void fix_x_y_dims(const float *X,
	const float *Y,
	const float *Out,
	int axis,
	uint16_t *x_dims,
	uint16_t *y_dims,
	uint16_t *p_x_dims,
	uint16_t *p_y_dims,
	uint16_t x_dims_size,
	uint16_t y_dims_size,
	int out_dim_size) {

	p_x_dims = (uint16_t *)calloc(out_dim_size, sizeof(uint16_t));
	p_y_dims = (uint16_t *)calloc(out_dim_size, sizeof(uint16_t));
	if (axis == -1) {
		int i_raw = 0;
		int i_new = out_dim_size - x_dims_size;
		for (; i_raw < x_dims_size; ++i_raw, ++i_new) {
			p_x_dims[i_new] = x_dims[i_raw];
		}
		i_raw = 0;
		i_new = out_dim_size - y_dims_size;
		for (; i_raw < y_dims_size; ++i_raw, ++i_new) {
			p_y_dims[i_new] = y_dims[i_raw];
		}
	}
	else {
		if (x_dims_size != out_dim_size) {
			if (y_dims_size != out_dim_size) {
				// LOG(FATAL) << "X/Y and OUT dim size mismatch";
			}
			else {
				// VLOG(4) << "Arguments broke API reference, for X.dims().size() is "
				//	"smaller and axis is set";
				for (int i = 0; i < out_dim_size; ++i) {
					p_y_dims[i] = y_dims[i];
				}
				for (int i = 0; i < x_dims_size; ++i) {
					p_x_dims[i + axis] = x_dims[i];
				}
			}
		}
		else {
			for (int i = 0; i < out_dim_size; ++i) {
				p_x_dims[i] = x_dims[i];
			}
			for (int i = 0; i < y_dims_size; ++i) {
				p_y_dims[i + axis] = y_dims[i];
			}
		}
	}
}
typedef struct  {
	int32_t z_num;

	int dim_size;
	int32_t continuous_length;
	enum BroadcastType broadcast_type;

	uint16_t* bcast_x_stride;
	uint16_t* bcast_y_stride;
	uint16_t* z_stride;
	uint16_t* element_id_stride;
}BatchElementWiseArg;
void Update(
	const float *x_data,
	const float *y_data,
	float *z_data,
	const uint16_t *x_dims,
	const uint16_t *y_dims,
	const uint16_t *z_dims,
	const uint16_t *x_stride,
	const uint16_t *y_stride,
	const uint16_t *z_stride,
	uint16_t dim_size,
	BatchElementWiseArg *batch_arg) {
	// arg checking
	batch_arg->broadcast_type = get_broadcast_type(x_dims, y_dims, z_dims, dim_size);


	if (x_stride[dim_size - 1] != 1 || y_stride[dim_size - 1] != 1 ||
		z_stride[dim_size - 1] != 1) {
		//LOG(FATAL) << "data are not stored continuously";
		return;
	}

	// generate element_id stride
	uint16_t* element_id_stride = (uint16_t*)calloc(dim_size, sizeof(uint16_t));
	for (int i = 0; i < dim_size; i++)
	{
		element_id_stride[i] = 1;
	}
	for (int i = dim_size - 2; i >= 0; --i) {
		element_id_stride[i] = z_dims[i + 1] * element_id_stride[i + 1];
	}

	// generate broadcast_stride
	uint16_t* bcast_x_stride = (uint16_t*)calloc(dim_size, sizeof(uint16_t));
	uint16_t* bcast_y_stride = (uint16_t*)calloc(dim_size, sizeof(uint16_t));
	memcpy(bcast_x_stride, x_stride, dim_size* sizeof(uint16_t));
	memcpy(bcast_y_stride, y_stride, dim_size * sizeof(uint16_t));
	int total_elem_num = 1;
	for (int i = 0; i < dim_size; ++i) {
		if (x_dims[i] == 1) {
			bcast_x_stride[i] = 0;
		}
		if (y_dims[i] == 1) {
			bcast_y_stride[i] = 0;
		}
		total_elem_num *= z_dims[i];
	}

	// get_continuous_length
	int64_t continuous_elem_num = z_dims[dim_size - 1];
	int end_pos = dim_size - 2;
	switch (batch_arg->broadcast_type) {
	case X_AS_CONTINUOUS: {
		while (end_pos >= 0 && y_dims[end_pos] == 1) {
			if (HasGapToNextDim(z_dims, z_stride, end_pos) ||
				HasGapToNextDim(x_dims, x_stride, end_pos)) {
				break;
			}
			continuous_elem_num *= z_dims[end_pos];
			--end_pos;
		}
		break;
	}
	case Y_AS_CONTINUOUS: {
		while (end_pos >= 0 && x_dims[end_pos] == 1) {
			if (HasGapToNextDim(z_dims, z_stride, end_pos) ||
				HasGapToNextDim(y_dims, y_stride, end_pos)) {
				break;
			}
			continuous_elem_num *= z_dims[end_pos];
			--end_pos;
		}
		break;
	}
	case BOTH_CONTINUOUS: {
		while (end_pos >= 0 && x_dims[end_pos] == y_dims[end_pos]) {
			if (HasGapToNextDim(z_dims, z_stride, end_pos) ||
				HasGapToNextDim(x_dims, x_stride, end_pos) ||
				HasGapToNextDim(y_dims, y_stride, end_pos)) {
				break;
			}
			continuous_elem_num *= z_dims[end_pos];
			--end_pos;
		}
		break;
	}

	default: {
		return;  // code should never goes to here
	}
	}
	batch_arg->z_num = total_elem_num;

	batch_arg->dim_size = dim_size;
	batch_arg->continuous_length = continuous_elem_num;

	batch_arg->bcast_x_stride = bcast_x_stride;
	batch_arg->bcast_y_stride = bcast_y_stride;
	batch_arg->z_stride = (uint16_t*)z_stride;
	batch_arg->element_id_stride = element_id_stride;
}

BatchElementWiseArg GenBatchElementWiseArg(const float *X,
	const float *Y,
	float *Out,
	int axis, uint16_t* x_dims, uint16_t* y_dims,uint16_t* z_dims, uint16_t x_dims_size,
	uint16_t y_dims_size, uint16_t out_dim_size) {
	BatchElementWiseArg batch_arg;
	/*uint16_t* o_x_dims = NULL;
	uint16_t* o_y_dims = NULL;
	fix_x_y_dims(X, Y, Out, axis, x_dims, y_dims, o_x_dims, o_y_dims, x_dims_size, y_dims_size, out_dim_size);*/

	// gen stride
	uint16_t* x_stride = (uint16_t*)malloc(out_dim_size*sizeof(uint16_t));
	uint16_t* y_stride = (uint16_t*)malloc(out_dim_size*sizeof(uint16_t));
	uint16_t* z_stride = (uint16_t*)malloc(out_dim_size*sizeof(uint16_t));
	for (int i = 0; i < out_dim_size; i++)
	{
		x_stride[i] = 1;
		y_stride[i] = 1;
		z_stride[i] = 1;
	}
	for (int i = out_dim_size - 2; i >= 0; --i) {
		x_stride[i] = x_stride[i + 1] * x_dims[i + 1];
		y_stride[i] = y_stride[i + 1] * y_dims[i + 1];
		z_stride[i] = z_stride[i + 1] * z_dims[i + 1];
	}

	Update(X,
		Y,
		Out,
		x_dims,
		y_dims,
		z_dims,
		x_stride,
		y_stride,
		z_stride,
		out_dim_size,
		&batch_arg);
	return batch_arg;
}

/**
 * Every element of some **FULL** tensor has its own logic id, ElemID2Offset
 * will convert this id to its memory offset
 * eg. given x.dims=[1,1,2,3] y.dims=[4,5,1,1],
 * then the full tensor's dim should be [4,5,2,3],
 * and, the element at [i,j,k,l] will get the
 * elem_id of `i*30 + j*6 + k*3 +l`
 * this elem_id works for all tensor X, Y and Z.
 */
int64_t ElemID2Offset(int64_t elem_id,
	const uint16_t* bcast_stride,
	const uint16_t* element_id_stride,
	int dim_size){
	int64_t ind = 0;
	int64_t offset = 0;
	for (int64_t i = 0; i < dim_size; ++i) {
		ind = elem_id / element_id_stride[i];
		offset += bcast_stride[i] * ind;
		elem_id -= (element_id_stride[i] * ind);
	}
	return offset;
}

float *AtBatch(float* x_data, int32_t batch_id, int32_t continuous_elem_num, uint16_t* bcast_x_stride, const uint16_t* element_id_stride, int dim_size) {
	return x_data +
		ElemID2Offset(batch_id * continuous_elem_num, bcast_x_stride, element_id_stride, dim_size);
}

static void X86CommonElementWise(float* x_data,
	float* y_data,
	float* out_data, BatchElementWiseArg batch_arg) {
	int batch_num = batch_arg.z_num / batch_arg.continuous_length;
	int range_length = batch_arg.continuous_length;
	switch (batch_arg.broadcast_type) {
	case (X_AS_CONTINUOUS): {
		for (int batch_id = 0; batch_id < batch_num; ++batch_id) {
			do_isa_elementwise(
				AtBatch(x_data, batch_id, batch_arg.continuous_length, batch_arg.bcast_x_stride, batch_arg.element_id_stride, batch_arg.dim_size),
				AtBatch(y_data, batch_id, batch_arg.continuous_length, batch_arg.bcast_y_stride, batch_arg.element_id_stride, batch_arg.dim_size),
				AtBatch(out_data, batch_id, batch_arg.continuous_length, batch_arg.z_stride, batch_arg.element_id_stride, batch_arg.dim_size),
				range_length, false, true);
		}
		break;
	}
	case (Y_AS_CONTINUOUS): {
		for (int batch_id = 0; batch_id < batch_num; ++batch_id) {
			do_isa_elementwise(
				AtBatch(x_data, batch_id, batch_arg.continuous_length, batch_arg.bcast_x_stride, batch_arg.element_id_stride, batch_arg.dim_size),
				AtBatch(y_data, batch_id, batch_arg.continuous_length, batch_arg.bcast_y_stride, batch_arg.element_id_stride, batch_arg.dim_size),
				AtBatch(out_data, batch_id, batch_arg.continuous_length, batch_arg.z_stride, batch_arg.element_id_stride, batch_arg.dim_size),
				range_length, true, false);
		}
		break;
	}
	case (BOTH_CONTINUOUS): {
		for (int batch_id = 0; batch_id < batch_num; ++batch_id) {
			do_isa_elementwise(
				AtBatch(x_data, batch_id, batch_arg.continuous_length, batch_arg.bcast_x_stride, batch_arg.element_id_stride, batch_arg.dim_size),
				AtBatch(y_data, batch_id, batch_arg.continuous_length, batch_arg.bcast_y_stride, batch_arg.element_id_stride, batch_arg.dim_size),
				AtBatch(out_data, batch_id, batch_arg.continuous_length, batch_arg.z_stride, batch_arg.element_id_stride, batch_arg.dim_size),
				range_length, false, false);
		}
		break;
	}
	default: {
		// LOG(FATAL) << "Un supported bcast type(isa)";
		break;
	}
	}
}

void elementwise_op(float* x_data,
	float* y_data,
	float* out_data, int axis, uint16_t* x_dims, uint16_t* y_dims, uint16_t* z_dims, uint16_t x_dims_size,
	uint16_t y_dims_size, uint16_t z_dims_size) {
	int pre, n, post;

	__m256 (*isa_op)(__m256, __m256) = &mul_ps_inline;
	float (*naive_op)(float, float) = &NaiveMul;
	if (same_dims(x_dims, y_dims, x_dims_size, y_dims_size)) {
		do_isa_elementwise(
			x_data, y_data, out_data, count(x_dims,0,x_dims_size), false, false);
	}
	else if (is_fast_broadcast(x_dims, y_dims, axis, x_dims_size,y_dims_size, &pre, &n, &post)) {
		Elementwise_Broadcast(
			x_data, y_data, out_data, pre, n, post, false);
	}
	else if (axis == -1 &&
		is_fast_broadcast(y_dims, x_dims, axis, y_dims_size, x_dims_size, &pre, &n, &post)) {
		Elementwise_Broadcast(
			x_data, y_data, out_data, pre, n, post, true);
	}
	else {
		BatchElementWiseArg batch_arg = GenBatchElementWiseArg(x_data, y_data, out_data, axis, x_dims, y_dims, z_dims, x_dims_size,y_dims_size, z_dims_size);
		X86CommonElementWise(x_data, y_data, out_data, batch_arg);
	}
}

void autox_elementwise_mul_ansi(float* x_data,
	float* y_data,
	float* out_data, int axis, uint16_t* x_dims, uint16_t* y_dims, uint16_t* z_dims, uint16_t x_dims_size,
	uint16_t y_dims_size, uint16_t z_dims_size) {

	__m256 (*isa_op)(__m256, __m256) = &mul_ps_inline;
	float (*naive_op)(float, float) = &NaiveMul;
	elementwise_op(x_data, y_data,out_data, axis,x_dims, y_dims,z_dims, x_dims_size,y_dims_size,z_dims_size);
}

void autox_elementwise_add_ansi(float* x_data,
	float* y_data,
	float* out_data, int axis, uint16_t* x_dims, uint16_t* y_dims, uint16_t* z_dims, uint16_t x_dims_size,
	uint16_t y_dims_size, uint16_t z_dims_size)
{
	__m256 (*isa_op)(__m256, __m256) = &add_ps_inline;
	float (*naive_op)(float, float) = &NaiveAdd;
	elementwise_op(x_data, y_data,out_data, axis,x_dims, y_dims,z_dims, x_dims_size,y_dims_size,z_dims_size);
}

void autox_fusion_elementwise_add_activation_ansi(float* x_data,
	float* y_data,
	float* out_data, int axis, uint16_t* x_dims, uint16_t* y_dims, uint16_t* z_dims, uint16_t x_dims_size,
	uint16_t y_dims_size, uint16_t z_dims_size)
{
	
}