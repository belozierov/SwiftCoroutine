//
//  CoFutureCancelTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 03.02.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class CoFutureCancelTests: XCTestCase {
    
    func testCancel() {
        let exp = expectation(description: #function)
        exp.expectedFulfillmentCount = 6
        let future = CoPromise<Bool>()
        func test() {
            future.whenComplete {
                if case .failure(let error as CoFutureError) = $0 {
                    XCTAssertEqual(error, .canceled)
                } else {
                    XCTFail()
                }
                exp.fulfill()
            }
            future.whenSuccess { _ in XCTFail() }
            future.whenCanceled { exp.fulfill() }
            future.whenFailure { error in
                if let error = error as? CoFutureError {
                    XCTAssertEqual(error, .canceled)
                } else {
                    XCTFail()
                }
                exp.fulfill()
            }
        }
        test()
        XCTAssertFalse(future.isCanceled)
        future.cancel()
        test()
        XCTAssertTrue(future.isCanceled)
        wait(for: [exp], timeout: 1)
    }
    
    func testCancel2() {
        let future = CoPromise<Int>()
        future.cancel()
        XCTAssertTrue(future.isCanceled)
    }
    
    func testCancelOnDeinit() {
        let exp = expectation(description: "testCancelOnDeinit")
        var promise: CoPromise<Int>! = CoPromise<Int>()
        promise.whenCanceled { exp.fulfill() }
        promise = nil
        wait(for: [exp], timeout: 1)
    }
    
    func testCancelParent() {
        let promise = CoPromise<Int>()
        let map = promise.map { $0 }
        map.cancel()
        XCTAssertTrue(promise.isCanceled)
        XCTAssertTrue(map.isCanceled)
    }
    
}
