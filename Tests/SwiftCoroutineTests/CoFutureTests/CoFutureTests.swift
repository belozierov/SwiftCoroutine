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
        let exp = expectation(description: "testResult2")
        exp.expectedFulfillmentCount = 4
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
    
}
