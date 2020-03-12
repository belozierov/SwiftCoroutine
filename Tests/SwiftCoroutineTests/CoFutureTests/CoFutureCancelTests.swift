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
        exp.expectedFulfillmentCount = 2
        let future = CoPromise<Bool>()
        future.whenCancelled(exp.fulfill)
        XCTAssertFalse(future.isCanceled)
        future.cancel()
        future.whenCancelled(exp.fulfill)
        XCTAssertTrue(future.isCanceled)
        wait(for: [exp], timeout: 1)
    }
    
    func testCancel2() {
        let future = CoPromise<Int>()
        let map = future.map { $0 + 1 }
        map.cancel()
        XCTAssertTrue(future.isCanceled)
        XCTAssertTrue(map.isCanceled)
    }
    
    func testCancel3() {
        let future = CoPromise<Int>()
        future.whenComplete { _ in }
        let map = future.map { $0 + 1 }
        map.cancel()
        XCTAssertFalse(future.isCanceled)
        XCTAssertTrue(map.isCanceled)
    }
    
    func testCancel4() {
        let future = CoPromise<Int>()
        future.whenComplete { _ in }
        let map = future.map { $0 + 1 }
        map.cancel()
        XCTAssertTrue(future.isCanceled)
        XCTAssertTrue(map.isCanceled)
    }
    
    func testCancel5() {
        let future = CoPromise<Int>()
        future.whenComplete { _ in }
        let map = future.map { $0 + 1 }
        let map2 = map.map { $0 + 1 }
        map.cancel()
        XCTAssertFalse(future.isCanceled)
        XCTAssertTrue(map.isCanceled)
        XCTAssertTrue(map2.isCanceled)
    }
    
}
