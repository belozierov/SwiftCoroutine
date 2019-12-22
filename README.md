# SwiftCoroutine

Many languages such as Kotlin, JavaScript, Go, Rust, C ++, and others already have [coroutines](https://en.wikipedia.org/wiki/Coroutine) support, which makes the use of asynchronous code easier. Unfortunately, Apple is still behind on this feature. But this can be improved by a framework without the need to change the language.

This is the first implementation of [coroutines](https://en.wikipedia.org/wiki/Coroutine) for Swift with macOS and iOS support of 64-bit systems (since support for 32-bit systems is no longer really relevant). The stackful coroutine approach is used because it has a minimal context switching overhead, high performance, and is best suited for implementation as a third-party framework.

The framework is fully integrated with [Dispatch](https://developer.apple.com/documentation/DISPATCH) making it intuitive to use. It is built on the [Futures and Promises](https://ru.wikipedia.org/wiki/Futures_and_promises) concept that facilitate the creation of extra extensions you might need. In addition, the framework can be easily used with the new [Combine](https://developer.apple.com/documentation/combine) framework and its army of various [Publishers](https://developer.apple.com/documentation/combine/publisher) and additional operators.

You can find some API similarity to the Kotlin coroutines, thanks to my friends Android developers who have constantly advised me on how it works.

### Usage

```swift

//Main thread

//If coroutine is started with default parameters on the main thread,
//it will also run on the main DispatchQueue
coroutine {
    //your custom extension that returns CoFuture<Data>
    let future = URLSession.shared.getData(with: imageURL)
    
    //await result that suspends coroutine and doesn't block the thread
    let data = try future.await()
    
    //coroutine is performed on the main thread, that's why we can set the image in UIImageView
    self.imageView.image = UIImage(data: data)
}
```

### Requirements

- iOS 11.0+ / macOS 10.13+
- Xcode 10.2+
- Swift 5+

### Installation

`SwiftCoroutine` is available through the [Swift Package Manager](https://swift.org/package-manager) for macOS and iOS.

## Working with SwiftCoroutine

### Futures and promises

Futures and promises are represented by the respective `CoFuture` class and its `CoPromise` and `CoLazyPromise` subclasses, which are generics to the type they return. They are thread-safe and have the support of the basic required functionality, including `await` mechanism, `onResult` on completion, and using the `transform` function you can build chains.

```swift
func makeSomeFuture() -> CoFuture<Int> {
    let promise = CoPromise<Int>()
    someAsyncFuncWithCompletion { int in
        promise.send(int)
    }
    return promise
}

let future = makeSomeFuture().transformValue { $0.description } 
future.onResult(queue: .global()) { result in
    //do some work with result of type Result<String, Error>
}
```

### Async/await

The framework includes extensions to DispatchQueue and global functions that allow you to execute code on a specific queue that returns `CoFuture`. You can wait for the result with the `await` function inside coroutine, which suspends it and does not block the thread, so you can do it, for example, on the main thread.

```swift
let future1: CoFuture<Int> = async(on: .global()) {
    sleep(2) //some work
    return 5
}

let future2: CoFuture<Int> = async(on: .global()) {
    sleep(3) //some work
    return 6
}

coroutine(on: .main) {
    let sum = try future1.await() + future2.await() //will await for 3 sec., doesn't block the thread
    self.label.text = "Sum is \(sum)"
}
```

### Coroutines

You can create coroutines with the DispatchQueue extension and the corresponding global functions. API is identical to `async` function, except that you can call await inside and suspend coroutines execution by resuming it when a `CoFuture` result is available. This makes it possible to write within the coroutines asynchronous code as synchronous. If global `coroutine` function is started with default parameters on the main thread, coroutine will run on the main DispatchQueue else on global.

This is a stackful coroutines, so each coroutine has its own stack, and after its completion, gets into the pool for reuse. If the system needs more RAM, the pool deinitializes all free coroutines and deallocates extra memory.

The framework also gives you access to the `Coroutine` class if you need more control or to write your own additional API.

```swift
coroutine {
     let coroutine: Coroutine! = .current //get current coroutine if needed
     someAsyncFuncWithCompletion {
         coroutine.resume() //manual resume outside coroutine
     }
     coroutine.suspend() //manual suspend inside coroutine
}
```

Also you can change DispatchQueue inside coroutine with the `setDispatcher` function.

```swift
coroutine(on: .global()) {
    //thread from global queue
    DispatchQueue.main.setDispatcher()
    //main thread
}
```

Or you can create coroutines with custom dispatchers.

```swift
let cor1 = Coroutine(dispatcher: { $0.block() })
let cor2 = Coroutine(dispatcher: { $0.block() })

cor1.start {
    //call 1
    cor1.suspend()
    //call 4
}
//call 2
cor2.start {
    //call 3
    cor1.resume()
    //call 5
}
//call 6
```

### Generators

The framework also includes the `Generator` class that allows yield values after each iteration similar to C#, Python, etc. [generators](https://en.wikipedia.org/wiki/Generator_(computer_programming)).

```swift
let generator = Generator<Int> { yield in
    for i in 0..<100 { yield(i) }
}
generator.next() //return 0
generator.next() //return 1
generator.next() //return 2
```

### Combine

Apple has recently introduced a new reactive programming framework that makes writing asynchronous code easier and includes a lot of convenient and common functionality. This framework includes the `await` extension for all publishers that allows combining reactive programming and coroutines for higher productivity.

```swift
let publisher = URLSession.shared.dataTaskPublisher(for: url).map(\.data)
coroutine {
    let data = try publisher.await()
    //do some work with data
}
```
