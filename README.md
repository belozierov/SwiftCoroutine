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
[![codecov](https://codecov.io/gh/belozierov/SwiftCoroutine/branch/master/graph/badge.svg)](https://codecov.io/gh/belozierov/SwiftCoroutine)
[![codebeat badge](https://codebeat.co/badges/4d86a744-7db9-4485-84d5-53390e1ae586)](https://codebeat.co/projects/github-com-belozierov-swiftcoroutine-master)

Many languages, such as Kotlin, JavaScript, Go, Rust, C++, and others, already have [coroutines](https://en.wikipedia.org/wiki/Coroutine) support that makes the use of asynchronous code easier. 
This feature is not yet supported in Swift, but this can be improved by a framework without the need to change the language.

This is the first implementation of [coroutines](https://en.wikipedia.org/wiki/Coroutine) for Swift with iOS, macOS and Linux support. They make the [async/await](https://en.wikipedia.org/wiki/Async/await) pattern implementation possible. In addition, the framework includes [futures](https://en.wikipedia.org/wiki/Futures_and_promises) and channels for more flexibility and ease of use. All this allows to do things that were not possible in Swift before.

### Motivation

Asynchronous programming is usually associated with callbacks. It is quite convenient until there are too many of them and they start nesting. Then it's called a **pyramid of doom** or even **callback hell**.

Another problem of asynchronous programming is **error handling**, because Swift's natural error handling mechanism cannot be used.

```swift
func postItem(item: Item) {
    preparePostAsync { token in
        submitPostAsync(token, item) { post in
            processPost(post)
        }
    }
}
```

#### What about Rx and other such frameworks?

There are many other frameworks that make it easy to use asynchronous code, such as Combine, RxSwift, PromiseKit and so on. They use other approaches that have some drawbacks:

- Similar to callbacks, you also need to create chained calls, that’s why you can’t normally use loops, exception handling, etc.
- Usually you need to learn a complex new API with hundreds of methods.
- Instead of working with the actual data, you need to operate with some wrappers all the time.
- Chaining of errors can be really complicated to handle.

```swift
func postItem(item: Item) {
    preparePostAsync() 
        .thenCompose { token in 
            submitPostAsync(token, item)
        }
        .thenAccept { post in 
            processPost(post)
        }
}

func preparePostAsync() -> Promise<Token> {
    // makes request an returns a promise that is completed later
    return promise 
}
```

### Async/await

The [async/await](https://en.wikipedia.org/wiki/Async/await) pattern is an alternative that allows an asynchronous, non-blocking function to be structured in a way similar to an ordinary synchronous function. 

It is already well-established in other programming languages and is an evolution in asynchronous programming. The implementation of this pattern is possible thanks to coroutines.

Let’s have a look at the example with coroutine inside of which `await()` suspends it and resumes when the result is available without blocking the thread.

```swift
func postItem(item: Item) {
    Coroutine.start {
        let token = awaitPreparePostAsync()
        let post = awaitPostAsync(token, item)
        processPost(post)
    }
}

func awaitPreparePostAsync() -> Token {
    //makes a request and suspends the coroutine
    return Coroutine.await { /* ... */ }
}
```
Here is the other example using `CoFuture`.

```swift
//executes coroutine on the main thread
DispatchQueue.main.startCoroutine {
    
    //extension that returns CoFuture<(data: Data, response: URLResponse)>
    let dataFuture = URLSession.shared.dataTaskFuture(for: imageURL)
    
    //await CoFuture result that suspends coroutine and doesn't block the thread
    let data: Data = try dataFuture.await().data

    //create UIImage from data or throw the error
    guard let image = UIImage(data: data) else { throw URLError(.cannotParseResponse) }
    
    //execute heavy task on global queue and await the result without blocking the thread
    let thumbnail: UIImage = try DispatchQueue.global().await { try image.makeThumbnail() }

    //set image in UIImageView on the main thread
    self.imageView.image = thumbnail
    
}
```

### Documentation

[API documentation](https://belozierov.github.io/SwiftCoroutine)

### Requirements

- iOS 11.0+ / macOS 10.13+ / Ubuntu
- Xcode 10.2+
- Swift 5+

### Installation

`SwiftCoroutine` is available through the [Swift Package Manager](https://swift.org/package-manager) for iOS, macOS and Linux.

## Working with SwiftCoroutine

### Coroutines

A [coroutines](https://en.wikipedia.org/wiki/Coroutine) is a computation that can be suspended and resumed at a later time without blocking a thread. Coroutines build upon regular functions and makes them more flexible.

Coroutines can be executed on any scheduler and switched between schedulers during execution.

#### Key benefits

- **Suspend instead of block**. The main advantage of coroutines is the ability to suspend their execution at some point without blocking a thread and resuming later on.
- **Fast context switching**. Switching between coroutines is much faster than switching between threads as it does not require the involvement of operating system.
- **Asynchronous code in synchronous manner**. The use of coroutines allows an asynchronous, non-blocking function to be structured in a manner similar to an ordinary synchronous function. And even though coroutines can run in multiple threads, your code will still look consistent and therefore easy to understand.

#### Usage

The coroutines API design is as minimalistic as possible. It consists of the `CoroutineScheduler` protocol that describes how to schedule coroutines (`DispatchQueue` already conforms it), and the `Coroutine` structure with utility methods. This API is enough to do amazing things.

The following example shows the usage of  `await()` inside a coroutine to wrap asynchronous calls.

```swift
//execute coroutine on the main thread
DispatchQueue.main.startCoroutine {
    
    //await URLSessionDataTask response without blocking the thread
    let (data, response, error) = Coroutine.await {
        URLSession.shared.dataTask(with: url, completionHandler: $0).resume()
    }
    
    . . . use response on the main thread . . . 
}
```

Here's how we can conform `NSManagedObjectContext` to `CoroutineScheduler` for launching coroutines on it.

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

    //execute request on the context without blocking the main thread
    let result = try context.await { try context.fetch(request) }
}
```

### Futures and Promises

A future is a read-only holder for a result that will be provided later and the promise is the provider of this result. They represent the eventual completion or failure of an asynchronous operation.

The [futures and promises](https://en.wikipedia.org/wiki/Futures_and_promises) approach itself has become an industry standart. It is a convenient mechanism to synchronize asynchronous code. But together with coroutines, it takes the usage of asynchronous code to the next level and has become a part of the async/await pattern. If coroutines are a skeleton, then futures and promises are its muscles.

#### Main features

- **Performance**. It is much faster than most of other futures and promises implementations.
- **Awaitable**. You can await the result inside the coroutine.
- **Cancellable**. You can cancel the whole chain as well as handle it and complete the related actions.

#### Usage

Futures and promises are represented by the corresponding `CoFuture` class and its `CoPromise` subclass. 

```swift
//wraps some async func with CoFuture
func makeIntFuture() -> CoFuture<Int> {
    let promise = CoPromise<Int>()
    someAsyncFunc { int in
        promise.success(int)
    }
    return promise
}
```

It allows to start multiple tasks in parallel and synchronize them later with `await()`.

```swift
//create CoFuture<Int> that takes 2 sec. from the example above 
let future1: CoFuture<Int> = makeIntFuture()

//execute task on the global queue and returns CoFuture<Int> with future result
let future2: CoFuture<Int> = DispatchQueue.global().coroutineFuture {
    Coroutine.delay(.seconds(3)) //some work that takes 3 sec.
    return 6
}

//execute coroutine on the main thread
DispatchQueue.main.startCoroutine {
    let sum: Int = try future1.await() + future2.await() //will await for 3 sec.
    self.label.text = "Sum is \(sum)"
}
```

It's very easy to transform or compose `CoFuture`s into a new one.

```swift
let array: [CoFuture<Int>]

//create new CoFuture<Int> with sum of future results
let sum = CoFuture { try array.reduce(0) { try $0 + $1.await() } }
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
    let data: Data = try future.await()
}
```

### Channels

Futures and promises provide a convenient way to transfer a single value between coroutines. Channels provide a way to transfer a stream of values. Conceptually, a channel is similar to a queue that allows to suspend a coroutine on receive if it is empty, or on send if it is full.

This non-blocking primitive is widely used in such languages as Go and Kotlin, and it is another instrument that improves working with coroutines.

#### Usage

To create channels, use the `CoChannel` class.

```swift
//create a channel with a buffer which can store only one element
let channel = CoChannel<Int>(maxBufferSize: 1)

DispatchQueue.global().startCoroutine {
    for i in 0..<10 {
        //imitate some work
        Coroutine.delay(.seconds(1))
        //sends a value to the channel and suspends coroutine if its buffer is full
        try channel.awaitSend(i)
    }
    
    //close channel when all values are sent
    channel.close()
}

DispatchQueue.global().startCoroutine {
    //receives values until closed and suspends a coroutine if it's empty
    for i in channel {
        print("Receive", i)
    }
    
    print("Done")
}
```
