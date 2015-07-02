package utility;

import java.util.Collection;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.RecursiveTask;

/**
 * Utilities for parallel computing in loops over independent tasks. This class
 * provides convenient methods for parallel processing of tasks that involve
 * loops over indices, in which computations for different indices are
 * independent.
 * <p>
 * As a simple example, consider the following function that squares floats in
 * one array and stores the results in a second array.
 * 
 * <pre>
 * <code>
 * static void sqr(float[] a, float[] b) {
 *   int n = a.length;
 *   for (int i=0; i&lt;n; ++i)
 *     b[i] = a[i]*a[i];
 * }
 * </code>
 * </pre>
 * 
 * A serial version of a similar function for 2D arrays is:
 * 
 * <pre>
 * <code>
 * static void sqrSerial(float[][] a, float[][] b) 
 * {
 *   int n = a.length;
 *   for (int i=0; i&lt;n; ++i) {
 *     sqr(a[i],b[i]);
 * }
 * </code>
 * </pre>
 * 
 * Using this class, the parallel version for 2D arrays is:
 * 
 * <pre>
 * <code>
 * static void sqrParallel(final float[][] a, final float[][] b) {
 *   int n = a.length;
 *   Parallel.loop(n,new Parallel.LoopInt() {
 *     public void compute(int i) {
 *       sqr(a[i],b[i]);
 *     }
 *   });
 * }
 * </code>
 * </pre>
 * 
 * In the parallel version, the method {@code compute} defined by the interface
 * {@code LoopInt} will be called n times for different indices i in the range
 * [0,n-1]. The order of indices is both indeterminant and irrelevant because
 * the computation for each index i is independent. The arrays a and b are
 * declared final as required for use in the implementation of {@code LoopInt}.
 * <p>
 * Note: because the method {@code loop} and interface {@code LoopInt} are
 * static members of this class, we can omit the class name prefix
 * {@code Parallel} if we first import these names with
 * 
 * <pre>
 * <code>
 * import static edu.mines.jtk.util.Parallel.*;
 * </code>
 * </pre>
 * 
 * A similar method facilitates tasks that reduce a sequence of indexed values
 * to one or more values. For example, given the following method:
 * 
 * <pre>
 * <code>
 * static float sum(float[] a) {
 *   int n = a.length;
 *   float s = 0.0f;
 *   for (int i=0; i&lt;n; ++i)
 *     s += a[i];
 *   return s;
 * }
 * </code>
 * </pre>
 * 
 * serial and parallel versions for 2D arrays may be written as:
 * 
 * <pre>
 * <code>
 * static float sumSerial(float[][] a) {
 *   int n = a.length;
 *   float s = 0.0f;
 *   for (int i=0; i&lt;n; ++i)
 *     s += sum(a[i]);
 *   return s;
 * }
 * </code>
 * </pre>
 * 
 * and
 * 
 * <pre>
 * <code>
 * static float sumParallel(final float[][] a) {
 *   int n = a.length;
 *   return Parallel.reduce(n,new Parallel.ReduceInt&lt;Float&gt;() {
 *     public Float compute(int i) {
 *       return sum(a[i]);
 *     }
 *     public Float combine(Float s1, Float s2) {
 *       return s1+s2;
 *     }
 *   });
 * }
 * </code>
 * </pre>
 * 
 * In the parallel version, we implement the interface {@code ReduceInt} with
 * two methods, one to {@code compute} sums of array elements and another to
 * {@code combine} two such sums together. The same pattern works for other
 * reduce operations. For example, with similar functions we could compute
 * minimum and maximum values (in a single reduce) for any indexed sequence of
 * values.
 * <p>
 * More general loops are supported, and are equivalent to the following serial
 * code:
 * 
 * <pre>
 * <code>
 * for (int i=begin; i&lt;end; i+=step)
 *   // some computation that depends on i
 * </code>
 * </pre>
 * 
 * The methods loop and reduce require that begin is less than end and that step
 * is positive. The requirement that begin is less than end ensures that reduce
 * is always well-defined. The requirement that step is positive ensures that
 * the loop terminates.
 * <p>
 * Static methods loop and reduce submit tasks to a fork-join framework that
 * maintains a pool of threads shared by all users of these methods. These
 * methods recursively split tasks so that disjoint sets of indices are
 * processed in parallel by different threads.
 * <p>
 * In addition to the three loop parameters begin, end, and step, a fourth
 * parameter chunk may be specified. This chunk parameter is a threshold for
 * splitting tasks so that they can be performed in parallel. If a range of
 * indices to be processed is smaller than the chunk size, or if too many tasks
 * have already been queued for processing, then the indices are processed
 * serially. Otherwise, the range is split into two parts for processing by new
 * tasks. If specified, the chunk size is a lower bound; the number of indices
 * processed serially will never be lower, but may be higher, than a specified
 * chunk size. The default chunk size is one.
 * <p>
 * The default chunk size is often sufficient, because the test for an excess
 * number of queued tasks prevents tasks from being split needlessly. This test
 * is especially useful when parallel loops are nested, as when looping over
 * elements of multi-dimensional arrays.
 * <p>
 * For example, an implementation of the method {@code sqrParallel} for 3D
 * arrays could simply call the 2D version listed above. Tasks will naturally
 * tend to be split for outer loops, but not inner loops, thereby reducing
 * overhead, time spent splitting and queueing tasks.
 * <p>
 * Reference: A Java Fork/Join Framework, by Doug Lea, describes the framework
 * used to implement this class. This framework will be part of JDK 7.
 * 
 * @author Dave Hale, Colorado School of Mines
 * @version 2010.11.23
 */
public class Parallel
{

	/** A loop body that computes something for an int index. */
	public interface LoopInt
	{

		/**
		 * Computes for the specified loop index.
		 * 
		 * @param i
		 *            loop index.
		 */
		public void compute(int i);
	}

	/** A loop body that computes and returns a value for an int index. */
	public interface ReduceInt<V>
	{

		/**
		 * Returns a value computed for the specified loop index.
		 * 
		 * @param i
		 *            loop index.
		 * @return the computed value.
		 */
		public V compute(int i);

		/**
		 * Returns the combination of two specified values.
		 * 
		 * @param v1
		 *            a value.
		 * @param v2
		 *            a value.
		 * @return the combined value.
		 */
		public V combine(V v1, V v2);
	}

	/**
	 * A wrapper for objects that are not thread-safe. Such objects have methods
	 * that cannot safely be executed concurrently in multiple threads. To use
	 * an unsafe object within a parallel computation, first construct an
	 * instance of this wrapper. Then, within the compute method, get the unsafe
	 * object; if null, construct and set a new unsafe object in this wrapper,
	 * before using the unsafe object to perform the computation. This pattern
	 * ensures that each thread computes using a distinct unsafe object. For
	 * example,
	 * 
	 * <pre>
	 * <code>
	 * final Parallel.Unsafe&lt;Worker&gt; nts = new Parallel.Unsafe&lt;Worker&gt;();
	 * Parallel.loop(count,new Parallel.LoopInt() {
	 *   public void compute(int i) {
	 *     Worker w = nts.get(); // get worker for the current thread
	 *     if (w==null) nts.set(w=new Worker()); // if null, make one
	 *     w.work(); // the method work need not be thread-safe
	 *   }
	 * });
	 * </code>
	 * </pre>
	 * 
	 * This wrapper is most useful when (1) the cost of constructing an unsafe
	 * object is high, relative to the cost of each call to compute, and (2) the
	 * number of threads calling compute is significantly lower than the total
	 * number of such calls. Otherwise, if either of these conditions is false,
	 * then simply construct a new unsafe object within the compute method.
	 * <p>
	 * This wrapper works much like the Java standard class ThreadLocal, except
	 * that an object within this wrapper can be garbage-collected before its
	 * thread dies. This difference is important because fork-join worker
	 * threads are pooled and will typically die only when a program ends.
	 */
	public static class Unsafe<T>
	{

		/**
		 * Constructs a wrapper for objects that are not thread-safe.
		 */
		public Unsafe()
		{
			int initialCapacity = 16; // the default initial capacity
			float loadFactor = 0.5f; // huge numbers of threads are unlikely
			int concurrencyLevel = 2 * _pool.getParallelism();
			_map = new ConcurrentHashMap<Thread, T>(initialCapacity,
				loadFactor, concurrencyLevel);
		}

		/**
		 * Gets the object in this wrapper for the current thread.
		 * 
		 * @return the object; null, of not yet set for the current thread.
		 */
		public T get()
		{
			return _map.get(Thread.currentThread());
		}

		/**
		 * Sets the object in this wrapper for the current thread.
		 * 
		 * @param object
		 *            the object.
		 */
		public void set(T object)
		{
			_map.put(Thread.currentThread(), object);
		}

		/**
		 * Returns a collection of all unsafe objects in this wrapper. This
		 * method is useful only after parallel loops have ended.
		 * 
		 * @return the collection of unsafe objects.
		 */
		public Collection<T> getAll()
		{
			return _map.values();
		}

		private final ConcurrentHashMap<Thread, T> _map;
	}

	/**
	 * Performs a loop <code>for (int i=0; i&lt;end; ++i)</code>.
	 * 
	 * @param end
	 *            the end index (not included) for the loop.
	 * @param body
	 *            the loop body.
	 */
	public static void loop(int end, LoopInt body)
	{
		loop(0, end, 1, 1, body);
	}

	/**
	 * Performs a loop <code>for (int i=begin; i&lt;end; ++i)</code>.
	 * 
	 * @param begin
	 *            the begin index for the loop; must be less than end.
	 * @param end
	 *            the end index (not included) for the loop.
	 * @param body
	 *            the loop body.
	 */
	public static void loop(int begin, int end, LoopInt body)
	{
		loop(begin, end, 1, 1, body);
	}

	/**
	 * Performs a loop <code>for (int i=begin; i&lt;end; i+=step)</code>.
	 * 
	 * @param begin
	 *            the begin index for the loop; must be less than end.
	 * @param end
	 *            the end index (not included) for the loop.
	 * @param step
	 *            the index increment; must be positive.
	 * @param body
	 *            the loop body.
	 */
	public static void loop(int begin, int end, int step, LoopInt body)
	{
		loop(begin, end, step, 1, body);
	}

	/**
	 * Performs a loop <code>for (int i=begin; i&lt;end; i+=step)</code>.
	 * 
	 * @param begin
	 *            the begin index for the loop; must be less than end.
	 * @param end
	 *            the end index (not included) for the loop.
	 * @param step
	 *            the index increment; must be positive.
	 * @param chunk
	 *            the chunk size; must be positive.
	 * @param body
	 *            the loop body.
	 */
	public static void loop(int begin, int end, int step, int chunk,
		LoopInt body)
	{
		checkArgs(begin, end, step, chunk);
		if (_serial || end <= begin + chunk * step) {
			for (int i = begin; i < end; i += step) {
				body.compute(i);
			}
		}
		else {
			LoopIntAction task = new LoopIntAction(begin, end, step, chunk,
				body);
			if (LoopIntAction.inForkJoinPool()) {
				task.invoke();
			}
			else {
				_pool.invoke(task);
			}
		}
	}

	/**
	 * Performs a reduce <code>for (int i=0; i&lt;end; ++i)</code>.
	 * 
	 * @param end
	 *            the end index (not included) for the loop.
	 * @param body
	 *            the loop body.
	 * @return the computed value.
	 */
	public static <V> V reduce(int end, ReduceInt<V> body)
	{
		return reduce(0, end, 1, 1, body);
	}

	/**
	 * Performs a reduce <code>for (int i=begin; i&lt;end; ++i)</code>.
	 * 
	 * @param begin
	 *            the begin index for the loop; must be less than end.
	 * @param end
	 *            the end index (not included) for the loop.
	 * @param body
	 *            the loop body.
	 * @return the computed value.
	 */
	public static <V> V reduce(int begin, int end, ReduceInt<V> body)
	{
		return reduce(begin, end, 1, 1, body);
	}

	/**
	 * Performs a reduce <code>for (int i=begin; i&lt;end; i+=step)</code>.
	 * 
	 * @param begin
	 *            the begin index for the loop; must be less than end.
	 * @param end
	 *            the end index (not included) for the loop.
	 * @param step
	 *            the index increment; must be positive.
	 * @param body
	 *            the loop body.
	 * @return the computed value.
	 */
	public static <V> V reduce(int begin, int end, int step, ReduceInt<V> body)
	{
		return reduce(begin, end, step, 1, body);
	}

	/**
	 * Performs a reduce <code>for (int i=begin; i&lt;end; i+=step)</code>.
	 * 
	 * @param begin
	 *            the begin index for the loop; must be less than end.
	 * @param end
	 *            the end index (not included) for the loop.
	 * @param step
	 *            the index increment; must be positive.
	 * @param chunk
	 *            the chunk size; must be positive.
	 * @param body
	 *            the loop body.
	 * @return the computed value.
	 */
	public static <V> V reduce(int begin, int end, int step, int chunk,
		ReduceInt<V> body)
	{
		checkArgs(begin, end, step, chunk);
		if (_serial || end <= begin + chunk * step) {
			V v = body.compute(begin);
			for (int i = begin + step; i < end; i += step) {
				V vi = body.compute(i);
				v = body.combine(v, vi);
			}
			return v;
		}
		else {
			ReduceIntTask<V> task = new ReduceIntTask<V>(begin, end, step,
				chunk, body);
			if (ReduceIntTask.inForkJoinPool()) {
				return task.invoke();
			}
			else {
				return _pool.invoke(task);
			}
		}
	}

	/**
	 * Enables or disables parallel processing by all methods of this class. By
	 * default, parallel processing is enabled. If disabled, all tasks will be
	 * executed on the current thread.
	 * <p>
	 * <em>Setting this flag to false disables parallel processing for all
	 * users of this class.</em> This method should therefore be used for
	 * testing and benchmarking only.
	 * 
	 * @param parallel
	 *            true, for parallel processing; false, otherwise.
	 */
	public static void setParallel(boolean parallel)
	{
		_serial = !parallel;
	}

	// /////////////////////////////////////////////////////////////////////////
	// private

	// Implementation notes:
	// Each fork-join task below has a range of indices to be processed.
	// If the range is less than or equal to the chunk size, or if the
	// queue for the current thread holds too many tasks already, then
	// simply process the range on the current thread. Otherwise, split
	// the range into two parts that are approximately equal, ensuring
	// that the left part is at least as large as the right part. If the
	// right part is not empty, fork a new task. Then compute the left
	// part in the current thread, and, if necessary, join the right part.

	// Threshold for number of surplus queued tasks. Used below to
	// determine whether or not to split a task into two subtasks.
	private static final int NSQT = 6;

	// The pool shared by all fork-join tasks created through this class.
	private static ForkJoinPool _pool = new ForkJoinPool();

	// Serial flag; true for no parallel processing.
	private static boolean _serial = false;

	/**
	 * Checks loop arguments.
	 */
	private static void checkArgs(int begin, int end, int step, int chunk)
	{
		argument(begin < end, "begin<end");
		argument(step > 0, "step>0");
		argument(chunk > 0, "chunk>0");
	}

	public static void argument(boolean condition, String message)
	{
		if (!condition)
			throw new IllegalArgumentException("required condition: " + message);
	}

	/**
	 * Splits range [begin:end) into [begin:middle) and [middle:end). The
	 * returned middle index equals begin plus an integer multiple of step.
	 */
	private static int middle(int begin, int end, int step)
	{
		return begin + step + ((end - begin - 1) / 2) / step * step;
	}

	/**
	 * Fork-join task for parallel loop.
	 */
	private static class LoopIntAction
		extends RecursiveAction
	{
		LoopIntAction(int begin, int end, int step, int chunk, LoopInt body)
		{
			assert begin < end : "begin < end";
			_begin = begin;
			_end = end;
			_step = step;
			_chunk = chunk;
			_body = body;
		}

		@Override
		protected void compute()
		{
			if (_end <= _begin + _chunk * _step
				|| getSurplusQueuedTaskCount() > NSQT) {
				for (int i = _begin; i < _end; i += _step) {
					_body.compute(i);
				}
			}
			else {
				int middle = middle(_begin, _end, _step);
				LoopIntAction l = new LoopIntAction(_begin, middle, _step,
					_chunk, _body);
				LoopIntAction r = (middle < _end) ? new LoopIntAction(middle,
					_end, _step, _chunk, _body) : null;
				if (r != null)
					r.fork();
				l.compute();
				if (r != null)
					r.join();
			}
		}

		private final int _begin, _end, _step, _chunk;
		private final LoopInt _body;
	}

	/**
	 * Fork-join task for parallel reduce.
	 */
	private static class ReduceIntTask<V>
		extends RecursiveTask<V>
	{
		ReduceIntTask(int begin, int end, int step, int chunk, ReduceInt<V> body)
		{
			assert begin < end : "begin < end";
			_begin = begin;
			_end = end;
			_step = step;
			_chunk = chunk;
			_body = body;
		}

		@Override
		protected V compute()
		{
			if (_end <= _begin + _chunk * _step
				|| getSurplusQueuedTaskCount() > NSQT) {
				V v = _body.compute(_begin);
				for (int i = _begin + _step; i < _end; i += _step) {
					V vi = _body.compute(i);
					v = _body.combine(v, vi);
				}
				return v;
			}
			else {
				int middle = middle(_begin, _end, _step);
				ReduceIntTask<V> l = new ReduceIntTask<V>(_begin, middle,
					_step, _chunk, _body);
				ReduceIntTask<V> r = (middle < _end) ? new ReduceIntTask<V>(
					middle, _end, _step, _chunk, _body) : null;
				if (r != null)
					r.fork();
				V v = l.compute();
				if (r != null)
					v = _body.combine(v, r.join());
				return v;
			}
		}

		private final int _begin, _end, _step, _chunk;
		private final ReduceInt<V> _body;
	}
}