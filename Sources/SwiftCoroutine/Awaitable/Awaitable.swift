////
////  Awaitable.swift
////  SwiftCoroutine
////
////  Created by Alex Belozierov on 30.01.2020.
////  Copyright © 2020 Alex Belozierov. All rights reserved.
////
//
//public protocol CoAwaitable {
//    
//    associatedtype Value
//    func whenComplete(_ callback: @escaping (Result<Value, Error>) -> Void)
//    
//}
//
//extension CoAwaitable {
//    
////    func await() throws -> Value {
////        try CoAwaiter().await(self)
////    }
//    
//}
//
//import Foundation
//
//struct FooAwaitable<Value> {
//    
//    typealias Completion = (Result<Value, Error>) -> Void
//    typealias Callback = (@escaping Completion) -> Result<Value, Error>?
//    
//    private let mutex: NSLock
//    private let callback: Callback
//    
//    init(mutex: NSLock, callback: @escaping Callback) {
//        self.mutex = mutex
//        self.callback = callback
//    }
//    
//    func await() throws -> Value {
//        let coroutine = try Coroutine.current()
//        var awaitResult: Result<Value, Error>!
//        mutex.lock()
//        if let result = callback({ result in
//            awaitResult = result
//            coroutine.resume()
//        }) {
//            mutex.unlock()
//            return try result.get()
//        }
//        coroutine.suspend(with: mutex.unlock)
//        return try awaitResult.get()
//    }
//    
//}
//
//
//final class CoAwaiter<Value> {
//    
//    private var result: Result<Value, Error>?
//    private var mutex: NSLock?
//    
//    init(result: Result<Value, Error>) {
//        self.result = result
//    }
//    
//    init(mutex: NSLock? = .init()) {
//        self.mutex = mutex
//    }
//    
//    func complete(with result: Result<Value, Error>) {
//        mutex?.lock()
//        if self.result != nil { return }
//        self.result = result
//        mutex?.unlock()
//        mutex = nil
//    }
//    
//    func await(mutex: NSLock?,
//               callback: @escaping ((Result<Value, Error>) -> Void) -> Void) throws -> Value {
//        let coroutine = try Coroutine.current()
//        var awaitResult: Result<Value, Error>!
//        callback { result in
//            awaitResult = result
//            if coroutine.state == .suspended { coroutine.resume() }
//        }
//        while true {
//            if let result = awaitResult {
//                return try result.get()
//            }
//            coroutine.suspend()
//        }
//    }
//    
//    func await<T: CoAwaitable>(_ awaitable: T) throws -> Value where T.Value == Value {
//        let coroutine = try Coroutine.current()
//        mutex?.lock()
//        if let result = result { return try result.get() }
//        awaitable.whenComplete { result in
//            self.complete(with: result)
//            if coroutine.state == .suspended { coroutine.resume() }
//        }
//        if let mutex = mutex {
//            coroutine.suspend(with: mutex.unlock)
//        } else {
//            coroutine.suspend()
//        }
//        return try result!.get()
//    }
//    
//}
