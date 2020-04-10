<!--
  Title: SwiftCoroutine
  Description: Swift coroutines for iOS and macOS.
  Author: belozierov
  Keywords: swift, coroutines, coroutine, async/await
  -->
  
![Swift Coroutine](../master/Sources/logo.png)

##

![macOS](https://github.com/belozierov/SwiftCoroutine/workflows/macOS/badge.svg?branch=master)
![Ubuntu](https://github.com/belozierov/SwiftCoroutine/workflows/Ubuntu/badge.svg?branch=master)
![codecov](https://codecov.io/gh/belozierov/SwiftCoroutine/branch/master/graph/badge.svg)

Many languages, such as Kotlin, JavaScript, Go, Rust, C++, and others, already have [coroutines](https://en.wikipedia.org/wiki/Coroutine) support that makes the use of asynchronous code easier. 
This feature is not yet supported in Swift, but this can be improved by a framework without the need to change the language.

This is the first implementation of [coroutines](https://en.wikipedia.org/wiki/Coroutine) for Swift with iOS, macOS and Linux support. They make the [async/await](https://en.wikipedia.org/wiki/Async/await) pattern implementation possible. In addition, the framework includes [futures and promises](https://en.wikipedia.org/wiki/Futures_and_promises) for more flexibility and ease of use. All this allows to do things that were not possible in Swift before.



### Usage

This is an example of a combined usage of coroutines with futures and promises.

```swift
//execute coroutine on the main thread
DispatchQueue.main.startCoroutine {

    //extension that returns CoFuture<(data: Data, response: URLResponse)>
    let dataFuture = URLSession.shared.dataTaskFuture(for: imageURL)
    
    //await result that suspends coroutine and doesn't block the thread
    let data = try dataFuture.await().data

    //create UIImage from data or throw the error
    guard let image = UIImage(data: data) else { throw URLError(.cannotParseResponse) }
    
    //execute heavy task on global queue and await the result without blocking the thread
    let thumbnail = DispatchQueue.global().await { image.makeThumbnail() }

    //set image in UIImageView on the main thread
    self.imageView.image = thumbnail
    
}
```

### Documentation

[API documentation](https://belozierov.github.io/SwiftCoroutine)

### Requirements

- iOS 11.0+ / macOS 10.13+ / Ubuntu 18.0+
- Xcode 10.2+
- Swift 5+

### Installation

`SwiftCoroutine` is available through the [Swift Package Manager](https://swift.org/package-manager) for iOS, macOS and Linux.

## Working with SwiftCoroutine

### Async/await

Asynchronous programming is usually associated with callbacks. It is quite convenient until there are too many of them and they start nesting. Then it's called **callback hell**.

The **async/await** pattern is an alternative. It is already well-established in other programming languages and is an evolution in asynchronous programming. The implementation of this pattern is possible thanks to coroutines. 

#### Key benefits
- **Suspend instead of block**. The main advantage of coroutines is the ability to suspend their execution at some point without blocking a thread and resuming later on.
- **Fast context switching**. Switching between coroutines is much faster than switching between threads as it does not require the involvement of operating system.
- **Asynchronous code in synchronous manner**. The use of coroutines allows an asynchronous, non-blocking function to be structured in a manner similar to an ordinary synchronous function. And even though coroutines can run in multiple threads, your code will still look consistent and therefore easy to understand.

The coroutines API design is as minimalistic as possible. It consists of the `CoroutineScheduler` protocol, which requires to implement only one method, and the `Coroutine` structure with utility methods. This API is enough to do amazing things.

The `CoroutineScheduler` protocol describes how to schedule tasks and as an extension you get the `startCoroutine()` method for executing coroutines on it, as well as the `await()` method for awaiting the result of the task (that is executed on your scheduler) inside the coroutine without blocking the thread. The framework includes the implementation of this protocol for `DispatchQueue`, but you can easily add it for other schedulers.

`Coroutine` has static utility methods for usage inside coroutines, including the `await()` method which suspends and resumes it on callback. It allows you to easily wrap asynchronous functions to deal with them as synchronous. 

#### Main features
- **Any scheduler**. You can use any scheduler to execute coroutines, including standard `DispatchQueue` or even `NSManagedObjectContext` and `MultiThreadedEventLoopGroup`.
- **Await instead of resume/suspend**. For convenience and safety, coroutines' resume/suspend has been replaced by await, which suspends it and resumes on callback.
- **Lock-free await**. Await is implemented using atomic variables. This makes it especially fast in cases where the result is already available.
- **Memory efficiency**. Contains a mechanism that allows to reuse stacks and, if necessary, effectively store their contents with minimal memory usage.
- **Create your own API**. Gives you a very flexible tool to create own powerful add-ons or easily integrate it with existing solutions.

The following example shows the usage of  `await()` inside a coroutine to manage asynchronous calls.

```swift
func awaitThumbnail(url: URL) throws -> UIImage {
    //await URLSessionDataTask response without blocking the thread
    let (data, _, error) = Coroutine.await {
        URLSession.shared.dataTask(with: url, completionHandler: $0).resume()
    }
    
    //parse UIImage or throw the error
    guard let image = data.flatMap(UIImage.init)
        else { throw error ?? URLError(.cannotParseResponse) }
    
    //execute heavy task on global queue and await its result
    return DispatchQueue.global().await { image.makeThumbnail() }
}

func setThumbnail(url: URL) {
    //execute coroutine on the main thread
    DispatchQueue.main.startCoroutine {
    
        //await image without blocking the thread
        let thumbnail = try? self.awaitThumbnail(url: url)
        
        //set image on the main thread
        self.imageView.image = thumbnail
    }
}
```

Here's how we can conform `NSManagedObjectContext` to `CoroutineScheduler`.

```swift
extension NSManagedObjectContext: CoroutineScheduler {

    func scheduleTask(_ task: @escaping () -> Void) {
        perform(task)
    }
    
}

//execute coroutine on the main thread
DispatchQueue.main.startCoroutine {
    let context: NSManagedObjectContext //context with privateQueueConcurrencyType
    let request: NSFetchRequest<Entity> //some complex request

    //execute request without blocking the main thread
    let result = try context.await { try context.fetch(request) }
}
```

### Futures and Promises

The futures and promises approach takes the usage of asynchronous code to the next level. It is a convenient mechanism to synchronize asynchronous code and has become a part of the async/await pattern. If coroutines are a skeleton, then futures and promises are its muscles.

Futures and promises are represented by the corresponding `CoFuture` class and its `CoPromise` subclass. `CoFuture` is a holder for a result that will be provided later.

#### Main features
- **Awaitable**. You can await the result inside the coroutine.
- **Cancellable**. You can cancel the whole chain as well as handle it and complete the related actions.
- **Combine-ready**. You can create `Publisher` from `CoFuture`, and vice versa make `CoFuture` a subscriber.

Here is an example of `URLSession` extension to creating `CoFuture` for `URLSessionDataTask`. The example of using it with coroutines and `await()` is provided [here](#Usage).

```swift
extension URLSession {

    typealias DataResponse = (data: Data, response: URLResponse)

    func dataTaskFuture(for urlRequest: URLRequest) -> CoFuture<DataResponse> {
        //create CoPromise that is a subclass of CoFuture for delivering the result
        let promise = CoPromise<DataResponse>()
    
        //create URLSessionDataTask
        let task = dataTask(with: urlRequest) {
            if let error = $2 {
                promise.fail(error)
            } else if let data = $0, let response = $1 {
                promise.success((data, response))
            } else {
                promise.fail(URLError(.badServerResponse))
            }
        }
        task.resume()
    
        //handle CoFuture canceling to cancel URLSessionDataTask
        promise.whenCanceled(task.cancel)
        
        return promise
    }
    
}
```

Also `CoFuture` allows to start multiple tasks in parallel and synchronize them later.

```swift
//execute task on the global queue and returns CoFuture<Int> with future result
let future1: CoFuture<Int> = DispatchQueue.global().coroutineFuture {
    Coroutine.delay(.seconds(2)) //some work that takes 2 sec.
    return 5
}

let future2: CoFuture<Int> = DispatchQueue.global().coroutineFuture {
    Coroutine.delay(.seconds(3)) //some work that takes 3 sec.
    return 6
}

//create new CoFuture with sum that will be completed in 3 sec.
let sumFuture = CoFuture { try future1.await() + future2.await() }
```

Apple has introduced a new reactive programming framework `Combine` that makes writing asynchronous code easier and includes a lot of convenient and common functionality. We can use it with coroutines by making `CoFuture` a subscriber and await its result.

```swift
//create Combine publisher
let publisher = URLSession.shared.dataTaskPublisher(for: url).map(\.data)

//execute coroutine on the main thread
DispatchQueue.main.startCoroutine {
    //subscribe CoFuture to publisher
    let future = publisher.subscribeCoFuture()
    
    //await data without blocking the thread
    let data = try future.await()
}
```
