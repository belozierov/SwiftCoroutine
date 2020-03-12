//
//  CoFutureTests2.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 26.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
import SwiftCoroutine
import AppKit
//@testable import SwiftCoroutine

class CoFutureTests3: XCTestCase {
    
    @inline(never) func emptyFunc() {
        
    }
    
    func testDispatch() {
        let group = DispatchGroup()
        measure {
            for _ in 0..<10_000 {
                group.enter()
                DispatchQueue.global().async(execute: group.leave)
            }
            group.wait()
        }
    }
    
    func testCoroutine() {
        let group = DispatchGroup()
        measure {
            for _ in 0..<10_000 {
                group.enter()
                CoroutineDispatcher.global.execute(group.leave)
            }
            group.wait()
        }
    }
    
    func testAb3() {
        measure {
            for _ in 0..<10_000 {
                let promise = CoPromise<Int>()
                let a = promise.map { $0 + 1 }
                a.whenComplete { _ in }
                a.whenComplete { _ in }
                a.map { $0 + 1 }.whenComplete { _ in }
                promise.success(0)
                let b = promise.map { $0 + 1 }
                b.whenComplete { _ in }
                b.whenComplete { _ in }
                b.map { $0 + 1 }.whenComplete { _ in }
            }
        }
    }
    
    func testAbc() {
        measure {
            for _ in 0..<10_000 {
                let promise = CoPromise<Int>()
                promise.map { $0 + 1 }
                    .always { _ in }
                    .always { _ in }
                    .map { $0 + 1 }
                    .whenComplete { _ in }
                promise.success(0)
                promise.map { $0 + 1 }
                    .always { _ in }
                    .always { _ in }
                    .map { $0 + 1 }
                    .whenComplete { _ in }
            }
        }
    }
    
    func testAb4() {
        measure {
            for _ in 0..<100 {
                let exp = expectation(description: "test")
                exp.expectedFulfillmentCount = 6
                let promise = CoPromise<Int>()
                let a = promise.map { $0 + 1 }
                a.whenComplete {
                    XCTAssertEqual($0, 1)
                    exp.fulfill()
                }
                a.whenComplete {
                    XCTAssertEqual($0, 1)
                    exp.fulfill()
                }
                a.map { $0 + 1 }.whenComplete {
                    XCTAssertEqual($0, 2)
                    exp.fulfill()
                }
                promise.success(0)
                let b = promise.map { $0 + 1 }
                b.whenComplete {
                    XCTAssertEqual($0, 1)
                    exp.fulfill()
                }
                b.whenComplete {
                    XCTAssertEqual($0, 1)
                    exp.fulfill()
                }
                b.map { $0 + 1 }.whenComplete {
                    XCTAssertEqual($0, 2)
                    exp.fulfill()
                }
                wait(for: [exp], timeout: 1)
            }
        }
    }
    
    func testDeinit() {
        var promise: CoPromise<Int>! = CoPromise()
        let handler = promise.always { _ in }
        promise = nil
        let handler2 = handler.always { _ in }
        print(promise)
    }
    
    func testFlatMap() {
        let exp = expectation(description: "test")
        let promise = CoPromise<Int>()
        promise.flatMap { value in
            self.future(with: value + 1, delay: 1)
        }.whenSuccess { value in
            XCTAssertEqual(value, 1)
            exp.fulfill()
        }
        promise.success(0)
        wait(for: [exp], timeout: 2)
    }
    
    func future<T>(with value: T, delay: Int) -> CoFuture<T> {
        let promise = CoPromise<T>()
        DispatchQueue.global().asyncAfter(deadline: .now() + 1) { promise.success(value) }
        return promise
    }
    
}
