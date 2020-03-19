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
    
    func testResult1() {
        let promise = CoPromise<Bool>()
        XCTAssertNil(promise.result)
        promise.success(true)
        promise.success(false)
        promise.cancel()
        XCTAssertEqual(promise.result, true)
    }
    
    func testResult2() {
        let future = CoFuture(result: .success(true))
        XCTAssertEqual(future.result, true)
        future.cancel()
        XCTAssertEqual(future.result, true)
    }
    
    func testMapOperators() {
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
    
    func testHashable() {
        let promise1 = CoPromise<Int>()
        let promise2 = CoPromise<Int>()
        XCTAssertNotEqual(promise1, promise2)
        XCTAssertNotEqual(promise1.hashValue, promise2.hashValue)
        XCTAssertEqual(promise1, promise1)
        XCTAssertEqual(promise1.hashValue, promise1.hashValue)
    }
    
}
