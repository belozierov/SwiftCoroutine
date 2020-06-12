//
//  CoFutureTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 31.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class CoFutureTests: XCTestCase {
    
    func testMeasure() {
        measure {
            DispatchQueue.concurrentPerform(iterations: 100_000) {
                let promise = CoPromise<Int>()
                let a = promise.map { $0 + 1 }
                a.whenComplete { _ in }
                a.whenComplete { _ in }
                a.map { $0 + 1 }.whenComplete { _ in }
                promise.success($0)
                let b = promise.map { $0 + 1 }
                b.whenComplete { _ in }
                b.whenComplete { _ in }
                b.map { $0 + 1 }.whenComplete { _ in }
            }
        }
    }
    
    func testResult1() {
        let promise = CoPromise<Bool>()
        XCTAssertNil(promise.result)
        promise.success(true)
        promise.success(false)
        promise.cancel()
        XCTAssertEqual(promise.result, true)
    }
    
    func testResult2() {
        let exp = expectation(description: "testResult2")
        exp.expectedFulfillmentCount = 6
        let future = CoFuture(result: .success(true))
        func test() {
            XCTAssertEqual(future.result, true)
            future.whenComplete {
                XCTAssertEqual(try? $0.get(), true)
                exp.fulfill()
            }
            future.whenSuccess {
                XCTAssertEqual($0, true)
                exp.fulfill()
            }
            future.whenComplete { exp.fulfill() }
            future.whenFailure { _ in XCTFail() }
            future.whenCanceled { XCTFail() }
        }
        test()
        future.cancel()
        test()
        wait(for: [exp], timeout: 1)
    }
    
    func testHashable() {
        let promise1 = CoPromise<Int>()
        let promise2 = CoPromise<Int>()
        XCTAssertNotEqual(promise1, promise2)
        XCTAssertNotEqual(promise1.hashValue, promise2.hashValue)
        XCTAssertEqual(promise1, promise1)
        XCTAssertEqual(promise1.hashValue, promise1.hashValue)
    }
    
    func testOptional() {
        let promise = CoPromise<Int?>()
        promise.success(nil)
        promise.success(1)
        switch promise.result {
        case .success(let value) where value == nil: break
        default: XCTFail()
        }
    }
    
    func testMap() {
        let exp = expectation(description: #function)
        exp.expectedFulfillmentCount = 2
        let promise = CoPromise<Int>()
        func test() {
            promise.map { $0 + 1 }.recover { _ in
                XCTFail()
                return 200
            }.mapResult { _ in
                Result<Int, Error>.failure(CoFutureError.canceled)
            }.recover {
                XCTAssertTrue($0 is CoFutureError)
                return 2
            }.whenSuccess {
                XCTAssertEqual($0, 2)
                exp.fulfill()
            }
        }
        test()
        promise.success(0)
        test()
        wait(for: [exp], timeout: 1)
    }
    
    func testCoroutineInit() {
        let date = Date()
        let exp = expectation(description: "testCoroutineInit")
        let future1 = DispatchQueue.global().coroutineFuture { () -> Int in
            try Coroutine.delay(.seconds(1))
            return 5
        }
        let future2 = DispatchQueue.global().coroutineFuture { () -> Int in
            try Coroutine.delay(.seconds(2))
            return 6
        }
        CoFuture { try future1.await() + future2.await() }.whenSuccess {
            XCTAssertEqual($0, 11)
            XCTAssertDuration(from: date, in: 2..<3)
            exp.fulfill()
        }
        wait(for: [exp], timeout: 3)
    }
    
    func testPromiseInit() {
        let exp = expectation(description: "testPromiseInit")
        func asyncFunc(_ callback: @escaping (Result<Int, Error>) -> Void) {
            DispatchQueue.global().asyncAfter(deadline: .now() + 1) {
                callback(.success(1))
            }
        }
        CoFuture(promise: asyncFunc).whenSuccess {
            XCTAssertEqual($0, 1)
            exp.fulfill()
        }
        wait(for: [exp], timeout: 5)
    }
    
}
