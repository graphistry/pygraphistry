# PyGraphistry Thread Safety Report

## Executive Summary

PyGraphistry's client implementation has limited thread safety. While not designed with concurrent access in mind, it provides a workable solution through the `client()` method that creates isolated instances. The global singleton pattern poses risks in multithreaded environments, but proper usage patterns can mitigate these issues.

## Key Findings

### 1. Thread Safety Status: **NOT Thread-Safe by Default**

The codebase lacks explicit thread safety mechanisms:
- No threading locks or synchronization primitives
- Mutable shared state without protection
- Global singleton instance shared across threads
- Class-level caches without thread safety

### 2. Concurrent Access Behavior

#### bind() Method (PlotterBase.py:1088)
- Creates shallow copies using `copy.copy()`
- Each call returns a new instance
- **Thread-safe for read operations**
- Shares underlying session state (potential race condition)

#### plot() Method (PlotterBase.py:1592)
- Includes token refresh mechanism before uploads
- `self._pygraphistry.refresh()` called at line 1693
- Handles token expiration gracefully
- **Partially thread-safe** due to refresh mechanism
- Still vulnerable to session state corruption

#### client() Method (pygraphistry.py:1717)
- Creates new GraphistryClient instances
- When `inherit=False`: Complete isolation
- When `inherit=True`: Shallow copy of session
- **Recommended approach for thread safety**

### 3. Session Management

ClientSession (client_session.py:26) contains mutable state:
- Authentication tokens
- API configuration
- Connection settings
- No thread synchronization

The `copy()` method (line 71) creates shallow copies:
- Dictionaries are copied
- Dataclasses are replaced
- Other objects share references

### 4. Shared Resources

#### Global Caches (PlotterBase.py)
```python
_pd_hash_to_arrow = weakref.WeakValueDictionary()
_cudf_hash_to_arrow = weakref.WeakValueDictionary()
_umap_param_to_g = weakref.WeakValueDictionary()
_feat_param_to_g = weakref.WeakValueDictionary()
```
These class-level caches are shared across all threads without protection.

#### Global Instance (pygraphistry.py:2480)
```python
PyGraphistry = GraphistryClient()
```
A global singleton instance accessible from all threads.

## Thread Safety Analysis

### Safe Operations
1. Creating isolated clients with `client(inherit=False)`
2. Read-only operations on immutable data
3. Independent plotting with separate client instances

### Unsafe Operations
1. Using the global `graphistry` instance from multiple threads
2. Sharing client instances across threads
3. Concurrent authentication/token refresh on same client
4. Modifying session configuration from multiple threads

## Recommendations

### For Thread-Safe Usage

1. **Always Use Isolated Clients**
   ```python
   # Thread function
   def worker_thread(user_id, data):
       # Create isolated client per thread
       client = graphistry.client(inherit=False)
       client.register(api=3, username=f'user_{user_id}', password='pass')
       
       # Bind and plot
       g = client.bind(source='src', destination='dst', nodes=data)
       url = g.plot()
       return url
   ```

2. **Avoid Global Instance**
   ```python
   # DON'T DO THIS
   import graphistry
   graphistry.register(...)  # Shared global state
   
   # DO THIS INSTEAD
   client = graphistry.client()
   client.register(...)  # Isolated state
   ```

3. **Thread Pool Pattern**
   ```python
   from concurrent.futures import ThreadPoolExecutor
   
   def process_dataset(dataset_id):
       # Each thread gets its own client
       client = graphistry.client(inherit=False)
       client.register(api=3, token=get_token())
       
       data = load_data(dataset_id)
       g = client.bind(source='src', destination='dst', nodes=data)
       return g.plot()
   
   with ThreadPoolExecutor(max_workers=10) as executor:
       futures = [executor.submit(process_dataset, i) for i in range(100)]
       results = [f.result() for f in futures]
   ```

### For Multiple Users

When supporting multiple users with different credentials:

```python
def create_user_client(username, password):
    """Create an isolated client for a specific user"""
    client = graphistry.client(inherit=False)
    client.register(api=3, username=username, password=password)
    return client

# Usage
client_alice = create_user_client('alice', 'pass1')
client_bob = create_user_client('bob', 'pass2')

# Each client maintains separate authentication
g_alice = client_alice.bind(source='src', destination='dst', nodes=alice_data)
g_bob = client_bob.bind(source='src', destination='dst', nodes=bob_data)

# Concurrent uploads work correctly
url_alice = g_alice.plot()
url_bob = g_bob.plot()
```

## Limitations

1. **Cache Sharing**: Even with isolated clients, the class-level caches are still shared
2. **Connection Pooling**: No built-in connection pooling for high-concurrency scenarios
3. **Resource Cleanup**: No explicit cleanup mechanism for client instances
4. **Memory Usage**: Each client maintains its own session state

## Conclusion

While PyGraphistry is not thread-safe by default, the `client(inherit=False)` pattern provides adequate isolation for most multithreaded use cases. The key is to:
- Never share client instances across threads
- Create new clients for each thread or user
- Avoid using the global instance in concurrent code
- Be aware of shared caches that may affect performance characteristics

For production multithreaded applications, consider implementing a client pool or using process-based parallelism for complete isolation.