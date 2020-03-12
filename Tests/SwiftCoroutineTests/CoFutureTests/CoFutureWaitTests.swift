//
//  CoFutureWaitTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 03.02.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
import Dispatch
@testable import SwiftCoroutine

class CoFutureWaitTests: XCTestCase {
    
    func testWait() {
        let promise = CoPromise<Int>()
        DispatchQueue.global().async {
            promise.success(1)
        }
        do {
            let result = try promise.wait()
            XCTAssertEqual(result, 1)
        } catch {
            XCTFail()
        }
    }
    
    func testWaitTimeout() {
        let promise = CoPromise<Int>()
        DispatchQueue.global().asyncAfter(deadline: .now() + .milliseconds(10)) {
            promise.success(1)
        }
        do {
            let result = try promise.wait(timeout: .now() + 1)
            XCTAssertEqual(result, 1)
        } catch {
            XCTFail()
        }
    }
    
    func testWaitTimeout2() {
        let promise = CoPromise<Int>()
        DispatchQueue.global().asyncAfter(deadline: .now() + 1) {
            promise.success(1)
        }
        do {
            _ = try promise.wait(timeout: .now() + .milliseconds(1))
            XCTFail()
        } catch let error as CoFutureError {
            XCTAssertEqual(error, .timeout)
        } catch {
             XCTFail()
        }
    }
    
    
}
