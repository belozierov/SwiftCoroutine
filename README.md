<!--
  Title: SwiftCoroutine
  Description: Swift coroutines for iOS and macOS.
  Author: belozierov
  Keywords: swift, coroutines, coroutine, async/await
  -->
  
![Swift Coroutine](../master/Sources/logo.png)

**Beta testing. Unit tests and documentation in progress. Possible minor changes in API.**
##
Many languages, such as Kotlin, JavaScript, Go, Rust, C++, and others, already have [coroutines](https://en.wikipedia.org/wiki/Coroutine) support that makes the use of asynchronous code easier. Unfortunately, Apple is still behind on this feature. But this can be improved by a framework without the need to change the language.

This is the first implementation of [coroutines](https://en.wikipedia.org/wiki/Coroutine) for Swift with macOS and iOS support. They make the [async/await](https://en.wikipedia.org/wiki/Async/await) pattern implementation possible. In addition, the framework includes [futures and promises](https://en.wikipedia.org/wiki/Futures_and_promises) for more flexibility and ease of use.

### Usage

This is an example of a combined usage of coroutines with futures and promises.

```swift
//execute coroutine on the main thread
//submit() returns CoFuture<Void>, thanks to which we can handle errors
CoroutineDispatcher.main.submit {

    //extension that returns CoFuture<(data: Data, response: URLResponse)>
    let dataFuture = URLSession.shared.dataTaskFuture(for: imageURL)
    
    //await result that suspends coroutine and doesn't block the thread
    let data = try dataFuture.await().data

    //create UIImage from data or throw the error
    guard let image = UIImage(data: data) else { throw URLError(.cannotParseResponse) }
    
    //execute heavy task on global queue and await the result without blocking the thread
    let thumbnail = try TaskScheduler.global.await { 
        image.makeThumbnail() //some method that returns UIImage
    }

    //coroutine is performed on the main thread, that's why we can set the image in UIImageView
    self.imageView.image = thumbnail
    
}.whenFailure { error in
    //here we can handle errors
}
```

### Requirements

- iOS 11.0+ / macOS 10.13+
- Xcode 10.2+
- Swift 5+

### Installation

`SwiftCoroutine` is available through the [Swift Package Manager](https://swift.org/package-manager) for macOS and iOS.

## Working with SwiftCoroutine

### Async/await

Asynchronous programming is usually associated with callbacks. It is quite convenient until there are too many of them and they start nesting. Then it's called **callback hell**.

The **async/await** pattern is an alternative. It is already well-established in other programming languages and is an evolution in asynchronous programming. The implementation of this pattern is possible thanks to coroutines. 

#### Key benefits
- **Suspend instead of block**. The main advantage of coroutines is the ability to suspend their execution at some point without blocking a thread and resuming later on.
- **Fast context switching**. Switching between coroutines is much faster than switching between threads as it does not require the involvement of operating system.
- **Asynchronous code in synchronous manner**. The use of coroutines allows an asynchronous, non-blocking function to be structured in a manner similar to an ordinary synchronous function. And even though coroutines can run in multiple threads, your code will still look consistent and therefore easy to understand.

You can execute tasks inside coroutines on `CoroutineDispatcher` just like you would do it on `DispatchQueue`. While `Coroutine.await()` allows you to wrap asynchronous functions to deal with them as synchronous. 

#### Main features
- **Any scheduler**. You can use any scheduler to execute coroutines wrapping it in `TashScheduler`, including standard `DispatchQueue` or even `NSManagedObjectContext` and `MultiThreadedEventLoopGroup`.
- **Memory efficiency**. Contains a mechanism that allows to reuse stacks and, if necessary, effectively store their contents.
- **Await instead of resume/suspend**. For convenience and safety, coroutines' resume/suspend has been replaced by await, which suspends it and resumes on callback.
- **Lock-free await**. Await is implemented using atomic variables. This makes it especially fast in cases where the result is already available.
- **Create your own API**. Gives you a very flexible tool to create your own add-ons or integrate with existing solutions.

The following example shows the usage of `await()` inside a coroutine to manage asynchronous calls.

```swift
func awaitThumbnail(url: URL) throws -> UIImage {
    //await URLSessionDataTask response without blocking the thread
    let (data, _, error) = try Coroutine.await {
        URLSession.shared.dataTask(with: url, completionHandler: $0).resume()
    }
    
    //parse UIImage or throw the error
    guard let image = data.flatMap(UIImage.init)
        else { throw error ?? URLError(.cannotParseResponse) }
    
    //execute heavy task on global queue and await its result
    return try TaskScheduler.global.await { image.makeThumbnail() }
}

func setThumbnail(url: URL) {
    //execute coroutine on the main thread
    CoroutineDispatcher.main.execute {
    
        //await image without blocking the thread
        let thumbnail = try? self.awaitThumbnail(url: url)
        
        //set image on the main thread
        self.imageView.image = thumbnail
    }
}
```

You can also wrap your scheduler in `TashScheduler` or conform to `TaskExecutor` protocol to await the result inside the coroutine, or create `CoFuture` with a deferred result. Here's how it might look for `CoreData`.

```swift
extension NSManagedObjectContext: TaskExecutor {
    public func execute(_ task: @escaping () -> Void) { perform(task) }
}

CoroutineDispatcher.main.execute {
    //some context with privateQueueConcurrencyType
    let context: NSManagedObjectContext
    //some complex request
    let request: NSFetchRequest<Entity>

    //execute fetch request without blocking the main thread
    let result = try context.await { try context.fetch(request) }
}
```

### Futures and Promises

The futures and promises approach takes the usage of asynchronous code to the next level. It is a convenient mechanism to synchronize asynchronous code and has become a part of the async/await pattern. If coroutines are a skeleton, then futures and promises are its muscles.

Futures and promises are represented by the corresponding `CoFuture` class and its `CoPromise` subclass. `CoFuture` is a holder for a result that will be provided later.

#### Main features
- **Best performance**. It is much faster than most of other futures and promises implementations.
- **Build chains**. With `flatMap()` and `map()`, you can create data dependencies via `CoFuture` chains.
- **Cancellable**. You can cancel the whole chain as well as handle it and complete the related actions.
- **Awaitable**. You can await the result inside the coroutine.
- **Combine-ready**. You can create `Publisher` from `CoFuture`, and vice versa make `CoFuture` a subscriber.

Here is an example of `URLSession` extension to creating `CoFuture` for `URLSessionDataTask`.

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

And this is how we can use the `URLSession` extension with chain of `CoFuture`s. The example of using it with coroutines and `await()` is provided [here](### Usage).

```swift
//create CoFuture with URLSessionDataTask response
URLSession.shared.dataTaskFuture(for: url)
    
    //parse data to Optional<UIImage>
    .map { UIImage(data: $0.data) }
    
    //if not nil return new CoFuture with heavy task that will execute it on global queue
    .flatMap { TaskScheduler.global.submit($0.makeThubnail) }
    
    //get Result<UIImage, Error> and set image on the main thread
    .whenComplete { result in
        DispatchQueue.main.execute { imageView.image = try? result.get() }
}
```

Also `CoFuture` allows to start multiple tasks in parallel and synchronize them later with `await()`.

```swift
//submits a given task to be executed on the global queue and returns CoFuture<Int> with deferred result
let future1 = TaskScheduler.global.submit { () -> Int in
    sleep(2) //some work
    return 5
}

let future2 = TaskScheduler.global.submit { () -> Int in
    sleep(3) //some work
    return 6
}

CoroutineDispatcher.main.execute {
    let sum = try future1.await() + future2.await() //will await for 3 sec.
    self.label.text = "Sum is \(sum)"
}
```

Apple has introduced a new reactive programming framework `Combine` that makes writing asynchronous code easier and includes a lot of convenient and common functionality. We can use it with coroutines by making `CoFuture` a subscriber and `await()` its result.

```swift
let publisher = URLSession.shared.dataTaskPublisher(for: url).map(\.data)

CoroutineDispatcher.main.execute {
    //subscribe CoFuture
    let future = publisher.subscribeCoFuture()
    
    //await data
    let data = try future.await()
    //do some work with data
}
```
